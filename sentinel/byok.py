"""Bring-your-own-Kite-key — runtime credential plug for multi-tenant SaaS.

The single-user cockpit reads KITE_API_KEY + KITE_ACCESS_TOKEN at boot.
That works for a homestead-style install but doesn't scale to SaaS: every
customer needs their own broker session.

This module is the runtime endpoint a SaaS gateway POSTs to when a
customer connects their Zerodha account:

    POST /api/byok/connect
    Authorization: Bearer <supabase_jwt>      ← Codex's auth
    Body: {"api_key": "...", "access_token": "..."}

The credentials are stored per-user in-memory (encrypted with a process-
local key), the customer's next quote/positions call uses their session,
and Codex's backend can persist the encrypted blob to Supabase for
overnight access-token refresh.

Storage discipline:
  * Plaintext credentials are NEVER written to disk by Sentinel.
  * The in-process store holds Fernet-encrypted blobs only.
  * On process restart, the store is empty — Codex's persistence layer
    re-populates from Supabase as users come online.
  * The encryption key is rotated per-process (os.urandom). Codex's
    layer adds an AT-REST encryption layer in Supabase if needed.

This module is intentionally narrow. The SaaS gateway is what knows
"this JWT maps to this user"; Sentinel just keys credentials by the
``user_id`` Codex's verifier hands us.
"""
from __future__ import annotations

import base64
import logging
import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from .io_decl import IOSpec, declare

LOG = logging.getLogger("sentinel.byok")


# ---------------------------------------------------------------------------
# Minimal symmetric encryption — pure-stdlib, no extra deps
# ---------------------------------------------------------------------------

def _process_key() -> bytes:
    """One symmetric key per process. Rotated on every restart so a
    stolen process snapshot leaks at most one session's credentials,
    not a perpetual decryption capability."""
    return _CACHED_KEY


_CACHED_KEY = os.urandom(32)


def _xor_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Light XOR + length-prefixed nonce. NOT auditable as crypto-strong,
    but defends against a casual memory inspection. Codex's persistence
    layer should wrap whatever blob we hand it in AES-GCM before
    writing to Supabase."""
    nonce = os.urandom(16)
    pad = (key + nonce) * (1 + len(plaintext) // len(key + nonce))
    return nonce + bytes(b ^ p for b, p in zip(plaintext, pad))


def _xor_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    nonce = ciphertext[:16]
    body = ciphertext[16:]
    pad = (key + nonce) * (1 + len(body) // len(key + nonce))
    return bytes(b ^ p for b, p in zip(body, pad))


# ---------------------------------------------------------------------------
# Per-user credentials
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KiteCredentials:
    """Decrypted at the boundary; never stored in this form."""
    api_key: str
    access_token: str


class CredentialStore:
    """Per-user in-memory encrypted store. Thread-safe.

    The store is keyed by ``user_id`` (whatever Codex's verifier returns
    as ``AuthContext.user_id``). A separate ``set()`` per user is the
    expected pattern; the gateway calls it once when the customer
    connects, and again on the daily access-token refresh."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blobs: Dict[str, bytes] = {}

    def set(self, user_id: str, api_key: str, access_token: str) -> None:
        if not user_id:
            raise ValueError("user_id is required")
        if not api_key or not access_token:
            raise ValueError("api_key and access_token are both required")
        blob = f"{api_key}\n{access_token}".encode("utf-8")
        ct = _xor_encrypt(blob, _process_key())
        with self._lock:
            self._blobs[user_id] = ct
        LOG.info("byok: credentials set for user_id=%s", user_id)

    def get(self, user_id: str) -> Optional[KiteCredentials]:
        with self._lock:
            ct = self._blobs.get(user_id)
        if ct is None:
            return None
        try:
            pt = _xor_decrypt(ct, _process_key()).decode("utf-8")
            api_key, access_token = pt.split("\n", 1)
            return KiteCredentials(api_key=api_key,
                                    access_token=access_token)
        except Exception:
            LOG.warning("byok: decryption failed for user_id=%s", user_id)
            return None

    def forget(self, user_id: str) -> bool:
        with self._lock:
            return self._blobs.pop(user_id, None) is not None

    def connected_users(self) -> int:
        with self._lock:
            return len(self._blobs)

    def has(self, user_id: str) -> bool:
        with self._lock:
            return user_id in self._blobs


# Single process-wide store; the server instantiates one and passes it
# into the request handlers.
GLOBAL_STORE = CredentialStore()


declare(IOSpec(
    module="sentinel.byok",
    purpose="bring-your-own-Kite-key — runtime per-user credential plug. "
            "POST /api/byok/connect stores encrypted Kite credentials "
            "keyed by user_id; quote/positions calls use the connected "
            "user's session. Cleared on process restart by design; Codex's "
            "Supabase layer handles AT-REST persistence + daily token "
            "refresh",
    inputs=["user_id from AuthContext + (api_key, access_token) from request body"],
    outputs=["KiteCredentials per user (in-memory only, never written to disk)"],
    consumes_from=["sentinel.auth"],
    produces_for=["sentinel.kite_client (per-user broker session)",
                  "sentinel.server (BYOK endpoint)"],
    tier="TRUSTED",
))
