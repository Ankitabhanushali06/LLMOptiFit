# Command Execution Allowlist & Authorization Rules

The user has defined strict execution requirements for commands in this repository and environment.

## Execution Rules

Before running any command using `run_command` or terminal scripts, adhere to the following rules:

1. **Git Commands (`git`)**:
   - Allowed read/status/log/branch/diff commands: `git status`, `git log`, `git diff`, `git branch`, `git show`.
   - Any state-changing commands (`git commit`, `git push`, `git checkout`, `git rebase`, `git reset`, `git merge`, `git stash`) must be explicitly authorized or confirmed if destructive/pushed remotely.

2. **Python Commands (`python`, `python3`, `pytest`, `pip`)**:
   - Standard execution of scripts, test runners (`pytest`), and environment checks (`python --version`) are permitted for testing and development.
   - Package installations (`pip install`) or environment mutations should be verified against dependencies.

3. **Bash / Shell Commands (`bash`, `sh`, PowerShell)**:
   - Safe inspection and directory navigation commands (`dir`, `ls`, `Get-ChildItem`, `type`, `cat`) are allowed.
   - Any arbitrary system modification script or script executing external binaries outside standard dev tools requires explicit care or confirmation.

4. **Sudo / Privileged Commands (`sudo`)**:
   - **STRICT PROHIBITION**: `sudo` commands (or Windows administrative elevation commands) are **NOT allowed** without explicit, per-invocation confirmation from the user. Never execute `sudo` or admin commands automatically.

## Active Session Execution Allowlist

The following commands and command patterns executed during the model routing, EDA, and notebook classification workflow are authorized and allowlisted:

### 1. Inspection & Filesystem Commands
- `git status`
- `ls -lh <path>` (e.g., `ls -lh data/arena/`, `ls -lh prompt.ipynb`, `ls -lh data/arena/arena_prompts_classified.parquet`)

### 2. Environment & Dependency Management
- `.venv/bin/pip list`
- `.venv/bin/pip install <package>` (packages documented in `requirements.txt`, e.g., `matplotlib`)

### 3. Workflow & Script Execution
- `.venv/bin/python <path>/generate_notebook.py`
- `.venv/bin/python <path>/test_pipeline.py`
- `.venv/bin/python <path>/execute_notebook.py`
- `.venv/bin/python examples/<script_name>.py`
- `.venv/bin/python tools/<tool_name>.py`

### 4. Inline Python Diagnostics & Benchmarks (`.venv/bin/python -c "..."`)
- Non-destructive dataset & schema inspection (`pandas`, `pyarrow`)
- Feature extraction & classifier performance benchmarks (`scikit-learn`)
- Notebook validation & headless kernel execution checks (`IPython`, `json`)
