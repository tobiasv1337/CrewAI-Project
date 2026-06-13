from __future__ import annotations

from typing import Optional

from .interfaces import CalculationResult
from .models import Module, ModuleState


_EPSILON = 1e-9


def _normalize_compact(value: str | None) -> str:
    return "".join((value or "").split()).lower()


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_grade(value: object) -> Optional[float]:
    try:
        grade = float(value)
    except (TypeError, ValueError):
        return None
    return grade if 1.0 <= grade <= 5.0 else None


def _discard_variant_by_key(
    result: CalculationResult,
    variant_key: Optional[str],
) -> Optional[dict[str, object]]:
    if not variant_key:
        return None
    for variant in list((result.calculation_details or {}).get("discard_variants") or []):
        if str(variant.get("key") or "") == variant_key:
            return dict(variant)
    return None


def discard_variant_options(result: CalculationResult) -> list[dict[str, str]]:
    variants = list((result.calculation_details or {}).get("discard_variants") or [])
    options: list[dict[str, str]] = []
    for variant in variants:
        key = str(variant.get("key") or "").strip()
        if not key:
            continue
        options.append(
            {
                "key": key,
                "label": str(variant.get("label") or key),
                "help": str(variant.get("help") or ""),
            }
        )
    return options if len(options) > 1 else []


def discard_variants_differ(result: CalculationResult) -> bool:
    variants = list((result.calculation_details or {}).get("discard_variants") or [])
    if len(variants) < 2:
        return False

    first = variants[0]
    baseline = (
        _as_float(first.get("value")),
        _as_float(first.get("raw_average")),
        _as_float(first.get("discarded_cp")),
        _as_float(first.get("counted_cp")),
    )
    for variant in variants[1:]:
        candidate = (
            _as_float(variant.get("value")),
            _as_float(variant.get("raw_average")),
            _as_float(variant.get("discarded_cp")),
            _as_float(variant.get("counted_cp")),
        )
        if any(abs(left - right) > _EPSILON for left, right in zip(baseline, candidate)):
            return True
    return False


def _module_from_variant_row(row: dict[str, object]) -> Module:
    grade = _as_grade(row.get("Grade"))
    credits = _as_float(row.get("Credits"), _as_float(row.get("CP"), 0.1))
    return Module(
        id=str(row.get("ID") or row.get("Module") or "discard-variant-module"),
        name=str(
            row.get("Module")
            or row.get("Name")
            or row.get("ID")
            or "Discard variant module"
        ),
        cp=max(credits, 0.1),
        grade=grade,
        area=str(row.get("Area") or "Unknown"),
        state=ModuleState.COMPLETED,
        is_graded=grade is not None,
    )


def _variant_counted_modules(
    result: CalculationResult,
    variant: dict[str, object],
    source_modules: list[Module],
) -> list[dict[str, object]]:
    source_by_id = {module.id: module for module in source_modules}
    details = result.calculation_details or {}
    protected_ids = set(details.get("protected_ids") or [])
    discarded_cp_by_id = {
        str(row.get("ID")): _as_float(row.get("Discarded credits"))
        for row in list(variant.get("modules") or [])
        if row.get("ID")
    }
    total_cp = _as_float(variant.get("counted_cp"))

    rows_by_id: dict[str, dict[str, object]] = {}

    def add_row(module_id: str, row: dict[str, object], module: Optional[Module] = None) -> None:
        full_cp = (
            module.cp
            if module is not None
            else _as_float(row.get("Credits"), _as_float(row.get("CP")))
        )
        discarded_cp = discarded_cp_by_id.get(module_id, 0.0)
        counted_cp = max(0.0, full_cp - discarded_cp)
        grade = module.effective_grade if module is not None else _as_grade(row.get("Grade"))
        if counted_cp <= _EPSILON or grade is None:
            return

        status = "Protected" if module_id in protected_ids else "Kept"
        if discarded_cp > _EPSILON:
            status = "Mixed"

        rows_by_id[module_id] = {
            "ID": module_id,
            "Module": (
                module.name
                if module is not None
                else str(row.get("Module") or row.get("Name") or module_id)
            ),
            "Area": module.area if module is not None else str(row.get("Area") or ""),
            "State": module.state.value if module is not None else str(row.get("State") or ""),
            "Credits": counted_cp,
            "Grade": grade,
            "Status": status,
        }

    for row in list(details.get("counted_modules") or []):
        module_id = str(row.get("ID") or "")
        if module_id:
            add_row(module_id, dict(row), source_by_id.get(module_id))

    for row in list(details.get("debug_candidates") or []):
        module_id = str(row.get("ID") or "")
        if module_id:
            add_row(module_id, dict(row), source_by_id.get(module_id))

    if total_cp <= 0:
        total_cp = sum(_as_float(row.get("Credits")) for row in rows_by_id.values())

    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (
            -_as_float(row.get("Credits")),
            _normalize_compact(str(row.get("Area"))),
            _normalize_compact(str(row.get("Module"))),
        ),
    )
    for row in rows:
        credits = _as_float(row.get("Credits"))
        grade = _as_float(row.get("Grade"))
        weight_percent = credits / total_cp * 100.0 if total_cp > _EPSILON else 0.0
        row["Weight %"] = weight_percent
        row["Grade contribution"] = grade * weight_percent / 100.0

    return rows


def _mark_selected_discard_variant(
    variants: list[object],
    selected_key: str,
) -> list[dict[str, object]]:
    marked: list[dict[str, object]] = []
    for raw_variant in variants:
        if not isinstance(raw_variant, dict):
            continue
        variant = dict(raw_variant)
        variant["status"] = (
            "Selected"
            if str(variant.get("key") or "") == selected_key
            else "Available"
        )
        marked.append(variant)
    return marked


def apply_discard_variant(
    result: CalculationResult,
    variant_key: Optional[str],
    source_modules: Optional[list[Module]] = None,
) -> CalculationResult:
    variant = _discard_variant_by_key(result, variant_key)
    if not variant:
        return result

    selected_key = str(variant.get("key") or "")
    if selected_key == "whole_modules":
        details = dict(result.calculation_details or {})
        details["selected_discard_variant"] = selected_key
        details["selected_discard_variant_label"] = variant.get("label")
        details["discard_variants"] = _mark_selected_discard_variant(
            list(details.get("discard_variants") or []),
            selected_key,
        )
        return result.model_copy(update={"calculation_details": details})

    source_modules = source_modules or []
    source_by_id = {module.id: module for module in source_modules}
    discarded_modules: list[Module] = []
    for row in list(variant.get("modules") or []):
        module_id = str(row.get("ID") or "")
        if not module_id or _as_float(row.get("Discarded credits")) <= _EPSILON:
            continue
        discarded_modules.append(
            source_by_id.get(module_id) or _module_from_variant_row(dict(row))
        )

    details = dict(result.calculation_details or {})
    details.update(
        {
            "raw_average": _as_float(variant.get("raw_average")),
            "selected_discard_variant": selected_key,
            "selected_discard_variant_label": variant.get("label"),
            "counted_modules": _variant_counted_modules(result, variant, source_modules),
            "discard_variants": _mark_selected_discard_variant(
                list(details.get("discard_variants") or []),
                selected_key,
            ),
        }
    )

    return result.model_copy(
        update={
            "final_grade": _as_float(variant.get("value"), result.final_grade),
            "graded_cp": _as_float(variant.get("counted_cp"), result.graded_cp),
            "discarded_cp": _as_float(variant.get("discarded_cp"), result.discarded_cp),
            "discarded_modules": discarded_modules,
            "calculation_details": details,
        }
    )
