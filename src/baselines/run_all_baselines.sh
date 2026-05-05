#!/bin/bash
echo "=== Running all baseline models ==="
echo

cd /app/src/baselines

python3 semgrep_baseline.py | tee results_semgrep.txt
python3 graphsage_baseline.py | tee results_graphsage.txt
python3 codebert_baseline.py | tee results_codebert.txt
python3 eval_tgat.py | tee results_tgat.txt

echo
echo "=== Baseline summary ==="
grep "📊" results_*.txt
