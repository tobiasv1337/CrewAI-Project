from __future__ import annotations

import secrets


def new_module_id() -> str:
    return f"mod_{secrets.token_hex(6)}"
