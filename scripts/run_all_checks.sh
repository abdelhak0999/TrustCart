#!/usr/bin/env bash
# Run every TrustCart check and write the results to test_report.txt (project root).
# Usage:  bash scripts/run_all_checks.sh
cd "$(dirname "$0")/.." || exit 1
PY=.venv-app/bin/python
OUT=test_report.txt
{
  echo "=== $(date) ==="
  echo; echo "### 1. verify_agents.py"; $PY scripts/verify_agents.py
  echo; echo "### 2. check_generalization.py"; $PY scripts/check_generalization.py
  echo; echo "### 3. VALID case (normal, templates)"; $PY scripts/simulate_case.py normal
  echo; echo "### 4. ALERT case (conceal, templates)"; $PY scripts/simulate_case.py conceal
  echo; echo "### 5. VALID case (normal, hybrid = LLM if a key is set)"; $PY scripts/simulate_case.py normal --mode hybrid
  echo; echo "### 6. ALERT case (conceal, hybrid)"; $PY scripts/simulate_case.py conceal --mode hybrid
  echo; echo "### 7. app.py compiles"; $PY -m py_compile app/app.py && echo "app.py OK"
  echo; echo "### 8. git status"; git status --short | head -60; git log --oneline | head -5
} > "$OUT" 2>&1
echo "Done -> $OUT"
