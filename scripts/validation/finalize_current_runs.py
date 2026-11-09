"""Finalize summaries and validation using three consistently rerun seeds."""

from pathlib import Path

import summarize_reproducibility as summary
import validate_outputs as validation


CURRENT_RUNS = {
    seed: summary.RESULTS_ROOT / f"raw/section_v_main_seed{seed}_e25"
    for seed in (42, 43, 44)
}


def main() -> None:
    summary.RUNS = dict(CURRENT_RUNS)
    validation.RUNS = dict(CURRENT_RUNS)
    summary.main()
    validation.main()


if __name__ == "__main__":
    main()
