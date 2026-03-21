from __future__ import annotations

try:
    from .ui import run_interface
except ImportError:
    from ui import run_interface


def main() -> None:
    run_interface()


if __name__ == "__main__":
    main()
