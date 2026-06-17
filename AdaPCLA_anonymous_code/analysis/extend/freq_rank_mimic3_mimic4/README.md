# Frequency–rank: MIMIC-III vs MIMIC-IV (Real)

建立长尾与两数据集的可比性：展示 MIMIC-III 与 MIMIC-IV 真实训练集的 code 频率–rank 曲线。

- **X 轴**：code rank（1 = 最常见）
- **Y 轴**：log(1 + frequency)

## 输出

- `output/freq_rank_mimic3_mimic4.png`：两条 Real 曲线

## 运行

```bash
cd extend/freq_rank_mimic3_mimic4
python3 plot_freq_rank_mimic3_mimic4.py
```

依赖：`fame/myfame/data/trainDataset.pkl`（MIMIC-III）、`fame/myfame/data2/trainDataset.pkl`（MIMIC-IV）
