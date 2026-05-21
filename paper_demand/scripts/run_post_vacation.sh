#!/usr/bin/env bash
# run_post_vacation.sh — exécute les 4 scripts d'analyse light sur les
# données collectées pendant les vacances.  Tout tourne en quelques
# minutes sur le VPS 4 GB.
#
# À lancer depuis le repo root :  bash paper_demand/scripts/run_post_vacation.sh
#
# Outputs : paper_demand/experiments/outputs/d6_*.csv, d7_*.csv,
#           d8_*.csv, d9_*.csv

set -u

REPO="/root/Recherche/imd-national-catalogue"
PY="$REPO/.venv/bin/python"
EXP="$REPO/paper_demand/experiments"
LOGS="$REPO/paper_demand/scripts/_vacation_logs"
mkdir -p "$LOGS"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
MASTER_LOG="$LOGS/_post_vacation.log"

run_one() {
  local name="$1"
  local script="$2"
  echo "" | tee -a "$MASTER_LOG"
  echo "════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
  echo "[$(stamp)] $name : $script" | tee -a "$MASTER_LOG"
  echo "════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
  t0=$(date +%s)
  "$PY" "$script" 2>&1 | tee -a "$MASTER_LOG"
  rc=${PIPESTATUS[0]}
  dt=$(( $(date +%s) - t0 ))
  echo "[$(stamp)] $name done in ${dt}s (rc=$rc)" | tee -a "$MASTER_LOG"
}

# Install scipy if missing (needed by d9)
"$PY" -c "import scipy" 2>/dev/null || "$PY" -m pip install -q scipy

cd "$REPO"

run_one "D6 — World atlas (183 cities)"       "$EXP/d6_intl_atlas.py"
run_one "D7 — Tier 1 descriptive (5 cities)"  "$EXP/d7_tier1_descriptive.py"
run_one "D8 — Polling quality (~43 cities)"   "$EXP/d8_polling_quality.py"
run_one "D9 — IMD components correlation"     "$EXP/d9_imd_components_corr.py"

echo "" | tee -a "$MASTER_LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$MASTER_LOG"
echo "[$(stamp)] ALL POST-VACATION SCRIPTS DONE" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Outputs in $EXP/outputs/:" | tee -a "$MASTER_LOG"
ls -1 "$EXP/outputs/" | grep -E "^d[6789]_" | sed 's/^/  /' | tee -a "$MASTER_LOG"
