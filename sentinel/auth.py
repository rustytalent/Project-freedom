"""Auth integration point — the plug Codex (Supabase + Google) connects to.

Sentinel doesn't own the auth flow. Codex's side ships the Google OAuth +
Supabase Auth pieces; their server (or their edge function) hands us a
signed token that proves the user's identity AND their current plan.

What Sentinel owns:

  * The CONTRACT — a Pydantic-ish dataclass ``AuthContext`` that every
    paid endpoint reads. Carries (user_id, email, plan, supabase_jwt).

  * A SINGLE point of override — ``verify_token(token: str) -> AuthContext``.
    Codex implements this for production (call Supabase's verifyJWT, look
    up the user, return their plan tier). The default implementation here
    parses an unsigned dev token so the cockpit keeps working in demo /
    local mode without any cloud dependencies.

  * A FastAPI dependency ``resolve_user`` that the existing ``require()``
    pipeline can use as a drop-in replacement for the simpler
    ``resolve_plan`` header-only flow.

How Codex plugs in (this is the WHOLE integration):

    # codex_auth.py (their file, not mine)
    from sentinel.auth import AuthContext, register_verifier
    import jwt, requests

    SUPABASE_JWT_SECRET = os.environ["SUPABASE_JWT_SECRET"]

    def supabase_verify(token: str) -> AuthContext:
        claims = jwt.decode(token, SUPABASE_JWT_SECRET,
                             algorithms=["HS256"], audience="authenticated")
        user_id = claims["sub"]
        email = claims["email"]
        # Codex's database lookup — plan tier lives in Supabase table
        plan = lookup_plan_from_supabase(user_id) or "RETAIL"
        return AuthContext(user_id=user_id, email=email,
                           plan=plan, supabase_jwt=token)

    register_verifier(supabase_verify)

That's it. Sentinel's code never imports Supabase, never reads a JWT
secret, never makes a network call. It just asks "verify this token,
give me a context" and uses what comes back.
"""
from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from .io_decl import IOSpec, declare


@dataclass(frozen=True)
class AuthContext:
    """Everything an endpoint needs to know about the caller."""
    user_id: str
    email: str = ""
    plan: str = "RETAIL"           # RETAIL / PRO / QUANT / FOUNDER
    supabase_jwt: str = ""          # raw token, in case downstream needs it
    is_demo: bool = False           # true when running without real auth

    @property
    def is_founder(self) -> bool:
        return self.plan.upper() == "FOUNDER"


# ---------------------------------------------------------------------------
# Verifier registry — Codex calls register_verifier() once at app boot
# ---------------------------------------------------------------------------

_verifier_lock = threading.Lock()
_verifier: Optional[Callable[[str], AuthContext]] = None


def register_verifier(fn: Callable[[str], AuthContext]) -> None:
    """Codex's production code calls this exactly once to plug in
    Supabase verification. After registration, every call to
    ``verify_token`` routes through ``fn``."""
    global _verifier
    with _verifier_lock:
        _verifier = fn


def reset_verifier() -> None:
    """Test helper — clears any registered verifier so the default
    dev-mode token parser is used."""
    global _verifier
    with _verifier_lock:
        _verifier = None


def verify_token(token: str) -> Optional[AuthContext]:
    """Returns an AuthContext if the token verifies, None otherwise.

    Production: routes through Codex's registered verifier.
    Dev/demo: parses a deliberately-unsigned token of the shape
    ``demo.<base64-json>.sig`` so the cockpit can run without Supabase.
    """
    if not token:
        return None
    with _verifier_lock:
        fn = _verifier
    if fn is not None:
        try:
            return fn(token)
        except Exception:
            return None
    return _dev_verify(token)


def _dev_verify(token: str) -> Optional[AuthContext]:
    """The dev fallback. Accepts:
      * 'demo'                    → demo founder ctx (no auth required)
      * 'demo.<b64json>.sig'      → unsigned JWT-shape with the b64 body
        carrying {sub, email, plan}
      * anything else             → None (treated as unauthenticated)
    """
    if token == "demo":
        return AuthContext(user_id="demo-founder",
                           email="founder@local",
                           plan="FOUNDER", is_demo=True)
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != "demo":
        return None
    try:
        body = base64.urlsafe_b64decode(parts[1] + "==").decode("utf-8")
        claims = json.loads(body)
    except Exception:
        return None
    return AuthContext(
        user_id=str(claims.get("sub", "demo")),
        email=str(claims.get("email", "")),
        plan=str(claims.get("plan", "RETAIL")).upper(),
        is_demo=True,
    )


def make_dev_token(plan: str = "FOUNDER", user_id: str = "dev",
                   email: str = "") -> str:
    """Build a dev token for local / test use. Codex's production users
    receive their tokens from Supabase Auth instead."""
    body = json.dumps({"sub": user_id, "email": email, "plan": plan})
    encoded = base64.urlsafe_b64encode(body.encode("utf-8")).rstrip(b"=").decode()
    return f"demo.{encoded}.sig"


# ---------------------------------------------------------------------------
# IO declaration
# ---------------------------------------------------------------------------

declare(IOSpec(
    module="sentinel.auth",
    purpose="auth integration plug — AuthContext contract + verify_token "
            "registry pattern; Codex's Supabase + Google Auth side calls "
            "register_verifier(fn) once at boot, Sentinel never imports "
            "Supabase or sees the JWT secret directly",
    inputs=["bearer token from Authorization header (Supabase JWT in "
            "production, dev token in local mode)"],
    outputs=["AuthContext(user_id, email, plan, supabase_jwt, is_demo)",
             "make_dev_token() for tests"],
    consumes_from=[],
    produces_for=["sentinel.server (every paid endpoint)",
                  "sentinel.saas (plan resolution)"],
    tier="TRUSTED",
))
