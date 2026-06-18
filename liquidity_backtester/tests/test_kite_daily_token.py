from pathlib import Path

import pytest

from scripts.kite_daily_token import load_env_file, parse_request_token, upsert_env_value


def test_parse_request_token_accepts_raw_token():
    assert parse_request_token("abc123") == "abc123"


def test_parse_request_token_accepts_redirect_url():
    url = "https://example.com/kite/callback?status=success&request_token=req_123&action=login"
    assert parse_request_token(url) == "req_123"


def test_parse_request_token_accepts_query_string():
    assert parse_request_token("?request_token=req_456&status=success") == "req_456"


def test_parse_request_token_rejects_url_without_token():
    with pytest.raises(ValueError, match="request_token"):
        parse_request_token("https://example.com/callback?status=success")


def test_load_env_file_handles_quotes_export_and_comments(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "KITE_API_KEY=abc",
                'KITE_API_SECRET="secret value"',
                "export KITE_ACCESS_TOKEN='token value'",
            ]
        ),
        encoding="utf-8",
    )

    assert load_env_file(env_file) == {
        "KITE_API_KEY": "abc",
        "KITE_API_SECRET": "secret value",
        "KITE_ACCESS_TOKEN": "token value",
    }


def test_upsert_env_value_updates_existing_and_preserves_other_lines(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# keep me\nKITE_API_KEY=old-key\nOTHER=value\n",
        encoding="utf-8",
    )

    upsert_env_value(env_file, "KITE_API_KEY", "new-key")
    upsert_env_value(env_file, "KITE_ACCESS_TOKEN", "token-123")

    assert env_file.read_text(encoding="utf-8").splitlines() == [
        "# keep me",
        "KITE_API_KEY=new-key",
        "OTHER=value",
        "KITE_ACCESS_TOKEN=token-123",
    ]
