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
