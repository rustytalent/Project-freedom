"""Zerodha Kite Connect login helper.

Run this ONCE every trading morning to generate a fresh access_token. Kite tokens expire
daily around midnight IST. Workflow:

  1. Set your API key and secret in env or pass on the command line:
        export KITE_API_KEY=your_api_key
        export KITE_API_SECRET=your_api_secret

  2. Run this script:
        python examples/zerodha_login.py

  3. It prints a Kite login URL. Open it in a browser, log in to your Zerodha account,
     authorise the app, and copy the `request_token` from the redirected URL.

  4. Paste the request_token back into the prompt.

  5. The script prints your access_token. Export it:
        export KITE_ACCESS_TOKEN=<paste here>

  6. Now you can run examples/live_run.py.

The access_token is valid until midnight IST. Repeat tomorrow.
"""
from __future__ import annotations
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("KITE_API_KEY", ""))
    ap.add_argument("--api-secret", default=os.environ.get("KITE_API_SECRET", ""))
    ap.add_argument("--request-token", default="",
                     help="If you've already done the OAuth redirect, paste request_token here")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("error: --api-key or KITE_API_KEY required")
    if not args.api_secret:
        sys.exit("error: --api-secret or KITE_API_SECRET required")

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        sys.exit("error: kiteconnect not installed. Run: pip install kiteconnect")

    kite = KiteConnect(api_key=args.api_key)

    if not args.request_token:
        print("\nStep 1 — open this URL in your browser, log in to Zerodha, authorise the app:\n")
        print(f"    {kite.login_url()}\n")
        print("After authorising, Kite redirects to your registered URL with ?request_token=...")
        print("Copy that token value here:\n")
        try:
            request_token = input("request_token: ").strip()
        except EOFError:
            sys.exit("\nno request_token provided")
        if not request_token:
            sys.exit("no request_token entered")
    else:
        request_token = args.request_token

    try:
        data = kite.generate_session(request_token, api_secret=args.api_secret)
    except Exception as e:
        sys.exit(f"error: failed to generate session: {e}")

    access_token = data.get("access_token")
    if not access_token:
        sys.exit("error: no access_token in response")

    print("\nSuccess. Export this in your shell before running live_run.py:\n")
    print(f"    export KITE_ACCESS_TOKEN={access_token}\n")
    print("User profile:")
    for k in ("user_name", "user_id", "broker", "email"):
        if k in data:
            print(f"    {k}: {data[k]}")
    print("\nAccess token expires at end of trading day. Re-run this tomorrow.")


if __name__ == "__main__":
    main()
