"""Daily Zerodha Kite access-token helper.

Kite access tokens expire daily. This script keeps the ritual explicit:

1. Read KITE_API_KEY and KITE_API_SECRET from args, environment, or .env.
2. Print the official Kite login URL.
3. Accept either the raw request_token or the full redirected URL.
4. Exchange it for KITE_ACCESS_TOKEN.
5. Optionally update .env without hardcoding secrets in source code.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a dotenv file."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = _strip_quotes(value)
    return values


def parse_request_token(text: str) -> str:
    """Accept either raw request_token or a full redirected Kite URL."""
    text = text.strip()
    if not text:
        raise ValueError("empty request token")

    if "request_token=" in text or "://" in text:
        parsed = urlparse(text)
        candidates = [parsed.query, text.lstrip("?")]
        for query_text in candidates:
            query = parse_qs(query_text)
            token = (query.get("request_token") or [""])[0].strip()
            if token:
                return token
        raise ValueError("redirect URL does not contain request_token")

    return text


def upsert_env_value(path: Path, key: str, value: str) -> None:
    """Update or append a dotenv value while preserving unrelated lines."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    replaced = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        body = stripped[len("export ") :].strip() if stripped.startswith("export ") else stripped
        if body.startswith(f"{key}="):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def _masked(value: str, keep: int = 4) -> str:
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}...{value[-keep:]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate today's Zerodha Kite access token and optionally write it to .env."
    )
    parser.add_argument("--env-file", default=".env", help="dotenv file to read/write")
    parser.add_argument("--api-key", default="", help="Kite API key; defaults to KITE_API_KEY/.env")
    parser.add_argument("--api-secret", default="", help="Kite API secret; defaults to KITE_API_SECRET/.env")
    parser.add_argument("--request-token", default="", help="Raw request_token from Kite redirect")
    parser.add_argument("--redirect-url", default="", help="Full redirected URL containing request_token")
    parser.add_argument("--write-env", action="store_true", help="Save KITE_ACCESS_TOKEN into --env-file")
    parser.add_argument("--print-login-url-only", action="store_true", help="Only print login URL and exit")
    parser.add_argument("--show-token", action="store_true", help="Print access token after generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env_file = Path(args.env_file)
    dotenv = load_env_file(env_file)

    api_key = args.api_key or os.environ.get("KITE_API_KEY", "") or dotenv.get("KITE_API_KEY", "")
    api_secret = (
        args.api_secret
        or os.environ.get("KITE_API_SECRET", "")
        or dotenv.get("KITE_API_SECRET", "")
    )
    if not api_key:
        sys.exit("error: missing KITE_API_KEY. Put it in .env or pass --api-key.")
    if not api_secret:
        sys.exit("error: missing KITE_API_SECRET. Put it in .env or pass --api-secret.")

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        sys.exit("error: kiteconnect is not installed. Run: .venv/bin/pip install kiteconnect")

    kite = KiteConnect(api_key=api_key)
    print("\nKite login URL:")
    print(kite.login_url())
    print("\nOpen it, log in, then copy the full redirected URL or request_token.\n")

    if args.print_login_url_only:
        return 0

    token_input = args.request_token or args.redirect_url
    if not token_input:
        try:
            token_input = input("Paste redirected URL or request_token: ")
        except EOFError:
            sys.exit("error: no request_token provided")

    try:
        request_token = parse_request_token(token_input)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    try:
        session = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:
        sys.exit(f"error: Kite session generation failed: {exc}")

    access_token = str(session.get("access_token") or "")
    if not access_token:
        sys.exit("error: Kite response did not contain access_token")

    if args.write_env:
        upsert_env_value(env_file, "KITE_API_KEY", api_key)
        upsert_env_value(env_file, "KITE_API_SECRET", api_secret)
        upsert_env_value(env_file, "KITE_ACCESS_TOKEN", access_token)
        print(f"Saved KITE_ACCESS_TOKEN to {env_file}")
    else:
        print("KITE_ACCESS_TOKEN generated. Export it for this shell:")
        print(f"export KITE_ACCESS_TOKEN={access_token}")

    if args.show_token:
        print(f"KITE_ACCESS_TOKEN={access_token}")
    else:
        print(f"Token: {_masked(access_token)}")

    print("\nKite profile:")
    for key in ("user_name", "user_id", "broker", "email"):
        if session.get(key):
            print(f"  {key}: {session[key]}")
    print("\nDone. Repeat this once every trading day before live mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
