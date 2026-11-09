# Working on the repository

Use Python 3.11 and install the dependency tier for the code you touch. Keep the main benchmark protocol separate from earlier experiments. Reference topology must never enter main training, prior/candidate construction, checkpoint selection or decoding.

Before proposing a change:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
git diff --check
```

For a pipeline change, also run the README smoke campaign, with and without evaluation when relevant. GPU changes require a hardware check; record the exact Torch build and device. New data adapters must document provenance, required source files, units, sampling, splits and generation parameters.

Commit source, small configuration, tests and usage documentation. Keep credentials, environment directories, assistant/editor files, upstream downloads, checkpoints, predictions, result tables and figures out of Git. Before pushing, inspect `git diff --cached --stat` and `git diff --cached --name-only`; do not use a blanket add on an unreviewed research directory.
