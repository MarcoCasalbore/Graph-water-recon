"""Regenerate all derived Section V artifacts from saved current runs."""

import build_section_v as build
import summarize_reproducibility as summary
import validate_outputs as validation


CURRENT_RUNS = {
    seed: build.RESULTS_ROOT / f"raw/section_v_main_seed{seed}_e25"
    for seed in (42, 43, 44)
}


def main() -> None:
    build.MAIN_RUN_DIR = CURRENT_RUNS[42]
    build.main()
    summary.RUNS = dict(CURRENT_RUNS)
    validation.RUNS = dict(CURRENT_RUNS)
    summary.main()
    validation.main()


if __name__ == "__main__":
    main()
