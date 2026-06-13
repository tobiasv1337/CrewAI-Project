from __future__ import annotations

import main as moses_cli


def test_cli_dispatches_single_moses_tool(monkeypatch, capsys):
    calls = []

    def fake_search_modules(query: str, max_results: int):
        calls.append((query, max_results))
        return "# fake module search"

    monkeypatch.setattr(moses_cli, "search_modules", fake_search_modules)

    exit_code = moses_cli.main(["search-modules", "Machine Learning", "--max-results", "4"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [("Machine Learning", 4)]
    assert "# fake module search" in captured.out


def test_cli_all_runs_every_moses_tool(monkeypatch, capsys):
    calls = []

    def record(name: str, output: str):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return output

        return fake

    monkeypatch.setattr(moses_cli, "search_modules", record("search_modules", "# module search"))
    monkeypatch.setattr(moses_cli, "get_module_details", record("get_module_details", "# module details"))
    monkeypatch.setattr(moses_cli, "get_module_catalogs", record("get_module_catalogs", "# module catalogs"))
    monkeypatch.setattr(moses_cli, "search_degree_programs", record("search_degree_programs", "# degree search"))
    monkeypatch.setattr(moses_cli, "get_degree_program_structure", record("get_degree_program_structure", "# degree structure"))
    monkeypatch.setattr(moses_cli, "get_degree_area_modules", record("get_degree_area_modules", "# degree area"))
    monkeypatch.setattr(moses_cli, "search_degree_modules", record("search_degree_modules", "# degree modules"))

    exit_code = moses_cli.main(
        [
            "all",
            "--module-query",
            "ML",
            "--module-number",
            "12345",
            "--module-version",
            "7",
            "--program-key",
            "TU Berlin - Test Program",
            "--degree-query",
            "Technische Informatik",
            "--area-query",
            "Wahlpflichtbereich",
            "--degree-module-query",
            "Dependable",
            "--max-results",
            "2",
            "--max-modules",
            "3",
            "--term",
            "SS 26",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert [call[0] for call in calls] == [
        "search_modules",
        "get_module_details",
        "get_module_catalogs",
        "search_degree_programs",
        "get_degree_program_structure",
        "get_degree_area_modules",
        "search_degree_modules",
    ]
    assert calls[0] == ("search_modules", ("ML",), {"max_results": 2})
    assert calls[1] == ("get_module_details", ("12345", 7), {"term": "SS 26"})
    assert calls[2] == (
        "get_module_catalogs",
        ("12345", 7),
        {"program_key": "TU Berlin - Test Program", "term": "SS 26"},
    )
    assert calls[3] == ("search_degree_programs", ("Technische Informatik",), {"max_results": 2})
    assert calls[4] == ("get_degree_program_structure", ("Technische Informatik",), {"term": "SS 26"})
    assert calls[5] == (
        "get_degree_area_modules",
        ("Technische Informatik", "Wahlpflichtbereich"),
        {"term": "SS 26", "max_modules": 3},
    )
    assert calls[6] == (
        "search_degree_modules",
        ("Technische Informatik", "Dependable"),
        {"term": "SS 26", "max_results": 2},
    )
    assert "# CLI smoke: search-modules" in captured.out
    assert "# CLI smoke: search-degree-modules" in captured.out
    assert "# degree modules" in captured.out


def test_cli_help_lists_all_moses_commands(capsys):
    parser = moses_cli.build_parser()

    try:
        parser.parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "search-modules" in captured.out
    assert "module-details" in captured.out
    assert "degree-area-modules" in captured.out
    assert "search-degree-modules" in captured.out
