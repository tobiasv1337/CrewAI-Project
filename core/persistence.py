from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from .module_origin import normalize_module_origin
from .models import CatalogAssignmentMode, Module, ModuleState
from .terms import term_sort_key


# ── Directory layout ──────────────────────────────────────────────────────────
DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
PROFILES_FILE = PROFILES_DIR / "profiles.json"

# Legacy flat file — used ONLY for the one-time migration on first startup.
_LEGACY_MODULES_FILE = DATA_DIR / "modules.json"

_STATE_ORDER = {
    ModuleState.COMPLETED: 0,
    ModuleState.IN_PROGRESS: 1,
    ModuleState.PLANNED: 2,
    ModuleState.POSSIBLE_CANDIDATE: 3,
}


# ── Profile model ─────────────────────────────────────────────────────────────

class ProfileRecord(BaseModel):
    model_config = {"validate_assignment": True}

    slug: str
    display_name: str
    is_primary: bool = False


# ── Slug helpers ──────────────────────────────────────────────────────────────

def _slug_from_name(name: str, existing_slugs: Optional[set] = None) -> str:
    """Generate a filesystem-safe slug from a display name with collision handling."""
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_") or "profile"
    slug = slug[:32]
    if existing_slugs is None or slug not in existing_slugs:
        return slug
    n = 2
    while f"{slug}_{n}" in existing_slugs:
        n += 1
    return f"{slug}_{n}"


# ── Path helpers ──────────────────────────────────────────────────────────────

def profile_dir(slug: str) -> Path:
    return PROFILES_DIR / slug


def modules_path(slug: str) -> Path:
    return profile_dir(slug) / "modules.json"


def _ensure_profile_dir(slug: str) -> None:
    profile_dir(slug).mkdir(parents=True, exist_ok=True)


# ── Profile persistence ───────────────────────────────────────────────────────

def load_profiles() -> List[ProfileRecord]:
    """Load the list of profiles from profiles.json.

    On the very first run (no profiles.json), performs a one-time migration:
    if the legacy ``data/modules.json`` exists, its contents are copied to the
    new primary profile directory.  Otherwise, an empty primary profile is
    created.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    if not PROFILES_FILE.exists():
        return _initialize_profiles()

    try:
        data = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
        profiles = [ProfileRecord(**item) for item in data]
        if not profiles:
            return _initialize_profiles()
        # Ensure exactly one primary.
        primaries = [p for p in profiles if p.is_primary]
        if not primaries:
            profiles[0].is_primary = True
            save_profiles(profiles)
        return profiles
    except Exception as exc:
        print(f"Error loading profiles: {exc}")
        return _initialize_profiles()


def _initialize_profiles() -> List[ProfileRecord]:
    """Bootstrap the profiles directory from scratch."""
    primary = ProfileRecord(slug="primary", display_name="Primary User", is_primary=True)
    _ensure_profile_dir(primary.slug)

    # One-time migration of legacy data/modules.json.
    target = modules_path(primary.slug)
    if _LEGACY_MODULES_FILE.exists() and not target.exists():
        shutil.copy2(str(_LEGACY_MODULES_FILE), str(target))
        print(f"Migrated {_LEGACY_MODULES_FILE} → {target}")
    elif not target.exists():
        target.write_text("[]", encoding="utf-8")

    profiles: List[ProfileRecord] = [primary]
    save_profiles(profiles)
    return profiles


def save_profiles(profiles: List[ProfileRecord]) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_FILE.write_text(
        json.dumps(
            [p.model_dump(mode="json") for p in profiles],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def add_profile(display_name: str, *, is_primary: bool = False) -> ProfileRecord:
    """Create a new named profile with an empty modules file."""
    profiles = load_profiles()
    existing_slugs = {p.slug for p in profiles}
    slug = _slug_from_name(display_name, existing_slugs)

    if is_primary:
        for p in profiles:
            p.is_primary = False

    new_profile = ProfileRecord(slug=slug, display_name=display_name, is_primary=is_primary)
    _ensure_profile_dir(slug)

    path = modules_path(slug)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")

    profiles.append(new_profile)
    save_profiles(profiles)
    return new_profile


def delete_profile(slug: str) -> None:
    """Delete a profile and its entire data directory.

    Raises ``ValueError`` if the slug belongs to the primary profile.
    """
    profiles = load_profiles()
    target = next((p for p in profiles if p.slug == slug), None)
    if target is None:
        return
    if target.is_primary:
        raise ValueError("Cannot delete the primary profile.")

    pdir = profile_dir(slug)
    if pdir.exists():
        shutil.rmtree(str(pdir))

    profiles = [p for p in profiles if p.slug != slug]
    save_profiles(profiles)


def set_primary_profile(slug: str) -> None:
    """Flip the is_primary flag so only *slug* is primary."""
    profiles = load_profiles()
    for p in profiles:
        p.is_primary = p.slug == slug
    save_profiles(profiles)


def rename_profile(slug: str, new_display_name: str) -> None:
    """Update the human-readable display name of a profile (slug unchanged)."""
    profiles = load_profiles()
    for p in profiles:
        if p.slug == slug:
            p.display_name = new_display_name
    save_profiles(profiles)


# ── Module persistence ────────────────────────────────────────────────────────

def load_modules(slug: str) -> List[Module]:
    """Load the module list for *slug*."""
    path = modules_path(slug)
    if not path.exists():
        _ensure_profile_dir(slug)
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        modules = [Module(**item) for item in data]
        _normalize_loaded_modules(modules)
        return modules
    except Exception as exc:
        print(f"Error loading modules for profile '{slug}': {exc}")
        return []


def save_modules(modules: List[Module], slug: str) -> None:
    """Persist the module list for *slug*."""
    _ensure_profile_dir(slug)
    for module in modules:
        normalize_module_origin(module)
    ordered = sorted(modules, key=_storage_sort_key)
    serialized = json.dumps(
        [m.model_dump(mode="json") for m in ordered],
        indent=4,
        ensure_ascii=False,
    )
    path = modules_path(slug)
    if path.exists() and path.read_text(encoding="utf-8") == serialized:
        return
    path.write_text(serialized, encoding="utf-8")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalize_loaded_modules(modules: List[Module]) -> None:
    if not modules:
        return

    try:
        from .registry import create_program
        from .providers.tu_berlin.moses import (
            _canonical_detail_url,
            infer_module_types_from_moses,
            infer_semester_span_from_moses,
        )
    except Exception:
        return

    strategies: dict = {}
    for module in modules:
        normalize_module_origin(module)
        if not module.program_key:
            continue
        if module.program_key not in strategies:
            try:
                strategies[module.program_key] = create_program(module.program_key)
            except Exception:
                strategies[module.program_key] = None
        strategy = strategies.get(module.program_key)
        if strategy is None:
            continue
        normalized_area = strategy.normalize_area(module.area)
        if normalized_area:
            module.area = normalized_area
        normalized_module_catalogs = [
            strategy.normalize_catalog(catalog) or catalog.strip()
            for catalog in module.catalogs
            if catalog.strip()
        ]
        if normalized_module_catalogs:
            module.catalogs = list(dict.fromkeys(normalized_module_catalogs))
        if module.moses is not None:
            if not module.module_types:
                module.module_types = infer_module_types_from_moses(module.moses)
            if module.semester_span <= 1:
                module.semester_span = infer_semester_span_from_moses(module.moses)
        if module.catalog_mode == CatalogAssignmentMode.AUTO:
            module.catalogs = []
        for reg in module.extra_registrations:
            if reg.program_key not in strategies:
                try:
                    strategies[reg.program_key] = create_program(reg.program_key)
                except Exception:
                    strategies[reg.program_key] = None
            reg_strategy = strategies.get(reg.program_key)
            if reg_strategy is None:
                continue
            normalized_reg_area = reg_strategy.normalize_area(reg.area)
            if normalized_reg_area:
                reg.area = normalized_reg_area
            if reg.catalog_mode == CatalogAssignmentMode.AUTO:
                reg.catalogs = []
            else:
                normalized_catalogs = [
                    reg_strategy.normalize_catalog(catalog) or catalog.strip()
                    for catalog in reg.catalogs
                    if catalog.strip()
                ]
                reg.catalogs = list(dict.fromkeys(normalized_catalogs))
        if module.moses_number and module.moses_version is not None:
            module.url = _canonical_detail_url(module.moses_number, module.moses_version)


def _storage_sort_key(module: Module) -> tuple:
    return (
        (module.program_key or "").lower(),
        _STATE_ORDER.get(module.state, 99),
        term_sort_key(module.term),
        (module.area or "").lower(),
        (module.name or "").lower(),
        module.id or "",
    )
