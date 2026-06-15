from __future__ import annotations

import base64
from urllib.parse import quote

from crew.isis_client import extract_mobile_wstoken


def test_extract_mobile_wstoken_accepts_moodlemobile_scheme_location():
    raw = "private-token:::0123456789abcdef0123456789abcdef:::site-id"
    token = quote(base64.b64encode(raw.encode("utf-8")).decode("ascii"))

    assert extract_mobile_wstoken(f"moodlemobile://token={token}") == "0123456789abcdef0123456789abcdef"


def test_isis_client_context_var_isolation():
    import contextvars
    from crew.isis_client import get_default_isis_client, set_default_isis_client, reset_default_isis_client, MoodleRestClient

    client1 = MoodleRestClient(wstoken="token1")
    client2 = MoodleRestClient(wstoken="token2")

    set_default_isis_client(client1)
    assert get_default_isis_client().wstoken == "token1"

    def run_in_context():
        reset_default_isis_client()
        set_default_isis_client(client2)
        assert get_default_isis_client().wstoken == "token2"

    ctx = contextvars.copy_context()
    ctx.run(run_in_context)

    assert get_default_isis_client().wstoken == "token1"
    reset_default_isis_client()
