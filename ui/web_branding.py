"""Brand the first HTTP response, before Streamlit sends page configuration.

Safari can keep the favicon from the initial HTML rather than the later
JavaScript update. Serve the app icon here as well as through set_page_config.
The HTML comes from the installed Streamlit bundle; its scripts stay intact.
"""

from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
import re

from starlette.responses import FileResponse, HTMLResponse
from starlette.routing import Route


ASSETS = Path(__file__).resolve().parents[1] / "assets"
ICON = ASSETS / "tu-berlin-icon.png"


@lru_cache(maxsize=1)
def initial_html() -> str:
    document = files("streamlit").joinpath("static/index.html").read_text(encoding="utf-8")
    # A new URL also avoids reusing the former immutable Streamlit favicon.
    version = sha256(ICON.read_bytes()).hexdigest()[:12]
    icons = (
        f'<link rel="icon" type="image/png" sizes="128x128" href="./favicon.png?v={version}">'
        f'<link rel="apple-touch-icon" sizes="128x128" href="./apple-touch-icon.png?v={version}">'
    )
    document = re.sub(
        r'<link\b(?=[^>]*\brel=[\"\'](?:shortcut )?icon[\"\'])[^>]*>',
        "",
        document,
        flags=re.IGNORECASE,
    )
    document = re.sub(r"<title>.*?</title>", "<title>Study Manager</title>", document, flags=re.DOTALL)
    return document.replace("</head>", icons + "</head>")


async def index(request):
    return HTMLResponse(initial_html(), headers={"Cache-Control": "no-cache"})


async def favicon(request):
    is_ico = request.url.path.endswith(".ico")
    return FileResponse(
        ICON.with_suffix(".ico") if is_ico else ICON,
        media_type="image/x-icon" if is_ico else "image/png",
        headers={"Cache-Control": "no-cache"},
    )


def branding_routes(base_path: str = "") -> list[Route]:
    prefix = "/" + base_path.strip("/") if base_path.strip("/") else ""
    return [
        Route(prefix + "/", index),
        Route(prefix + "/index.html", index),
        Route(prefix + "/favicon.png", favicon),
        Route(prefix + "/favicon.ico", favicon),
        Route(prefix + "/apple-touch-icon.png", favicon),
        Route(prefix + "/apple-touch-icon-precomposed.png", favicon),
    ]
