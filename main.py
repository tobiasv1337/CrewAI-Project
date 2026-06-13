from __future__ import annotations

from crew.tools.moses_tools import get_module_catalogs, get_module_details, search_modules


def main() -> None:
    print(search_modules("Machine Learning"))
    print("\n" + "=" * 80 + "\n")
    print(get_module_details("40966", 2))
    print("\n" + "=" * 80 + "\n")
    print(get_module_catalogs("40966", 2))


if __name__ == "__main__":
    main()
