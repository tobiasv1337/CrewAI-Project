"""Verify browser branding before a Streamlit session or JavaScript exists."""

from io import BytesIO
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from PIL import Image
import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient
from streamlit.web.server.starlette.starlette_static_routes import create_streamlit_static_assets_routes

from ui.web_branding import ICON, branding_routes


@pytest.mark.parametrize("prefix", ["", "/study"])
def test_initial_document_has_working_icons_and_streamlit_bundle(prefix):
    application = Starlette(routes=branding_routes(prefix) + create_streamlit_static_assets_routes(prefix))
    with TestClient(application) as client:
        response = client.get(prefix + "/?page=Dashboard&program_view=All")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        soup = BeautifulSoup(response.text, "html.parser")
        assert soup.title.string == "Study Manager"
        icons = soup.select('link[rel="icon"], link[rel="apple-touch-icon"]')
        assert len(icons) == 2
        for icon in icons:
            url = urljoin(str(response.url), icon["href"])
            assert urlsplit(url).query.startswith("v=")
            image = client.get(url)
            assert image.status_code == 200
            assert image.headers["content-type"] == "image/png"
            assert image.content == ICON.read_bytes()
        # Keep the installed frontend intact, including relative URLs under a base path.
        scripts = soup.select('script[src]')
        assert scripts
        for script in scripts:
            bundle = client.get(urljoin(str(response.url), script["src"]))
            assert bundle.status_code == 200
            assert "javascript" in bundle.headers["content-type"]


@pytest.mark.parametrize("path", ["favicon.png", "favicon.ico", "apple-touch-icon.png"])
def test_fallback_icons_are_real_images_without_immutable_cache(path):
    with TestClient(Starlette(routes=branding_routes())) as client:
        response = client.get("/" + path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        image = Image.open(BytesIO(response.content))
        assert image.format == ("ICO" if path.endswith(".ico") else "PNG")
        assert image.size == (128, 128)
        assert client.head("/" + path).headers["content-length"] == str(len(response.content))
