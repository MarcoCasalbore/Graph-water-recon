"""Placeholder for future inference entry points.

Inference is currently handled by the training/export scripts and the water
comparison utilities. Keep this file as the stable location for a future
standalone inference CLI.
"""


def main() -> None:
    raise SystemExit(
        "Standalone inference is not implemented yet. "
        "Use scripts/water/compare_graphs_epanet.py for learned graph inspection."
    )


if __name__ == "__main__":
    main()
