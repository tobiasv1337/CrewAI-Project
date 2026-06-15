from __future__ import annotations

import base64
from urllib.parse import quote

from crew.isis_client import extract_mobile_wstoken


def test_extract_mobile_wstoken_accepts_moodlemobile_scheme_location():
    raw = "private-token:::0123456789abcdef0123456789abcdef:::site-id"
    token = quote(base64.b64encode(raw.encode("utf-8")).decode("ascii"))

    assert extract_mobile_wstoken(f"moodlemobile://token={token}") == "0123456789abcdef0123456789abcdef"

