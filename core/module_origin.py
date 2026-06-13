from __future__ import annotations

from urllib.parse import urlparse

from .models import Module, ModuleSource


_INSTITUTION_BY_HOST = {
    "tu-berlin.de": "TU Berlin",
    "hu-berlin.de": "HU Berlin",
}


def is_moses_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    return host.endswith("tu-berlin.de") and "/moses/" in path


def infer_institution_from_url(url: str | None) -> str | None:
    if not url:
        return None
    host = (urlparse(url).netloc or "").lower()
    for suffix, institution in _INSTITUTION_BY_HOST.items():
        if host == suffix or host.endswith(f".{suffix}"):
            return institution
    return host or None


def infer_module_source(module: Module) -> ModuleSource:
    if module.moses_number and module.moses_version is not None:
        return ModuleSource.MOSES
    if module.moses is not None:
        return ModuleSource.MOSES
    if is_moses_url(module.url):
        return ModuleSource.MOSES
    if module.url:
        return ModuleSource.EXTERNAL
    return ModuleSource.MANUAL


def infer_module_institution(module: Module) -> str | None:
    if module.source == ModuleSource.MOSES:
        return "TU Berlin"
    inferred = infer_institution_from_url(module.url)
    if inferred:
        return inferred
    if module.source == ModuleSource.MANUAL:
        return "TU Berlin"
    return None


def normalize_module_origin(module: Module) -> None:
    source_missing = "source" not in module.model_fields_set
    institution_missing = "institution" not in module.model_fields_set or not module.institution

    if source_missing:
        module.source = infer_module_source(module)

    if institution_missing:
        inferred_institution = infer_module_institution(module)
        if inferred_institution:
            module.institution = inferred_institution
