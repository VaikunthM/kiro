---
inclusion: always
---

# This machine

- The terminal is Windows PowerShell. Bash syntax does not work here: no heredocs (`<<'EOF'`), no
  `&&` chains on older shells, no `export`. To run multi-line Python, write a `.py` file and run it.
- Python lives in the conda env `hackachusetts`. Run Python commands as
  `conda run -n hackachusetts python ...` (the terminal opens in `base`, which lacks the packages).
- Use paths relative to the workspace root; the terminal's working directory is the workspace root.
- Ask before installing packages into an environment.
