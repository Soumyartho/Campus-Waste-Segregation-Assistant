# CLAUDE.md

- Project: campus waste classifier with disposal recommendations. `PLAN.md` is the source of truth.
- Python 3.10+, PyTorch, type hints on public functions, short functions, docstrings on modules.
- All paths and hyperparameters come from YAML configs in `configs/`. No hard-coded values in code.
- Set seeds (Python, NumPy, PyTorch) with `waste.utils.set_seed` and call it everywhere.
- Never commit `data/`, `models/*.pt` or secrets. Keep `.gitignore` current.
- Ask before adding a dependency that is not in PLAN.md Section 4.
- Evaluate on validation for decisions and on test only for final numbers.
- After each phase: run `pytest` and `ruff check .`, update the README commands, and print a short summary.
- Commands to keep working: `make data`, `make train-baseline`, `make train`, `make eval`, `make app`, `make test`.
  On Windows without make, use `./tasks.ps1 <target>` with the same target names.
- The package lives in `src/waste` and is installed editable (`pip install -e .`).
- Windows: every script that builds a `DataLoader` must keep its entry point under `if __name__ == "__main__":`.
