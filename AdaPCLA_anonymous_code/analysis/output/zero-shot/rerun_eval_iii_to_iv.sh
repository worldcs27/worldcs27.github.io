#!/usr/bin/env bash
# 仅重跑 III→IV zero-shot 评估：清除 MyModel2 缓存，用映射后的数据重新训练 classifier，
# 再更新 zeroshot_table3.csv（III→IV 用新结果，IV→III 保留现有结果）。

set -e
ZERO_SHOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MYWORK="$(dirname "$(dirname "$ZERO_SHOT_DIR")")"
PCLA_ROOT="$(dirname "$MYWORK")"
FAME="$PCLA_ROOT/fame/myfame"
DATA_IV="$FAME/data2"
EVAL_PY="$FAME/evaluate/evaluate_synthetic_training.py"
OUT_DIR="$ZERO_SHOT_DIR/output"
EVAL_DIR_3="$OUT_DIR/eval_model3_to_iv"
EVAL_DIR_5="$OUT_DIR/eval_model5_to_iii"
MAPPED_PKL="$OUT_DIR/model3_to_iv_syn_mapped.pkl"

echo "=== 1. 清除 III→IV 评估目录下 MyModel2 的缓存（强制用 mapped 数据重训） ==="
rm -f "$EVAL_DIR_3"/syn_diag_MyModel2_*.pt
echo "已删除 $EVAL_DIR_3/syn_diag_MyModel2_*.pt"

if [[ ! -f "$MAPPED_PKL" ]]; then
  echo "错误: 未找到映射后的数据 $MAPPED_PKL，请先运行 ./run_zeroshot.sh 生成。" >&2
  exit 1
fi

echo ""
echo "=== 2. 仅运行 III→IV 下游评估（base=IV, synthetic=mapped pkl） ==="
python3 "$EVAL_PY" \
  --base_data_dir "$DATA_IV" \
  --mymodel2_path "$MAPPED_PKL" \
  --save_dir "$EVAL_DIR_3" \
  --sources MyModel2

echo ""
echo "=== 3. 用新 III→IV 结果 + 现有 IV→III 结果 重写 zeroshot_table3.csv ==="
python3 - "$OUT_DIR" "$EVAL_DIR_3" "$EVAL_DIR_5" << 'PY'
import csv
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
eval_dir_3 = Path(sys.argv[2])
eval_dir_5 = Path(sys.argv[3])

def parse_mean(csv_path, source="MyModel2"):
    accs, f1s, auprcs = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source") != source:
                continue
            a, f1, au = row.get("Accuracy"), row.get("F1 Score"), row.get("AUPRC")
            if a: accs.append(float(a))
            if f1: f1s.append(float(f1))
            if au: auprcs.append(float(au))
    return (
        sum(accs) / len(accs) if accs else 0.0,
        sum(f1s) / len(f1s) if f1s else 0.0,
        sum(auprcs) / len(auprcs) if auprcs else 0.0,
    )

csv3 = eval_dir_3 / "compare_real_halo_mymodel2.csv"
csv5 = eval_dir_5 / "compare_real_halo_mymodel2.csv"
if not csv3.exists():
    sys.exit(f"Expected {csv3} after eval")
acc_3, f1_3, auprc_3 = parse_mean(csv3)
rows = [{"target": "MIMIC-IV", "method": "AdaPCLA (III→IV zero-shot)", "Acc": acc_3, "F1": f1_3, "AUPRC": auprc_3}]
if csv5.exists():
    acc_5, f1_5, auprc_5 = parse_mean(csv5)
    rows.append({"target": "MIMIC-III", "method": "AdaPCLA (IV→III zero-shot)", "Acc": acc_5, "F1": f1_5, "AUPRC": auprc_5})

out_csv = out_dir / "zeroshot_table3.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["target", "method", "Acc", "F1", "AUPRC"])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {out_csv}")
print("III→IV:", f"Acc={acc_3:.4f} F1={f1_3:.4f} AUPRC={auprc_3:.4f}")
if csv5.exists():
    print("IV→III (unchanged):", f"Acc={acc_5:.4f} F1={f1_5:.4f} AUPRC={auprc_5:.4f}")
PY

echo ""
echo "Done. 请查看 output/zeroshot_table3.csv 和 output/eval_model3_to_iv/compare_real_halo_mymodel2.csv"
