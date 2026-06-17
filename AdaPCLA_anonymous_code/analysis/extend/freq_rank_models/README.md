# Frequency–rank: Real vs Models (方案 A)

多模型 frequency–rank 曲线叠在一起对比（MIMIC-IV）：Real、HALO、LSTM、GPT、AdaPCLA。

- **X 轴**：code rank（1 = 最常见）
- **Y 轴**：log(1 + frequency)
- 理想情况：AdaPCLA 曲线最接近 Real

## 输出

- `output/freq_rank_models_mimic4.png` / `freq_rank_models_mimic3.png`：全范围、log(rank) x 轴、多条曲线
- `output/freq_rank_models_mimic4_tail.png` / `freq_rank_models_mimic3_tail.png`：tail 专用图（rank ≥ 5000，线性 x，raw frequency y）

## 运行

```bash
cd extend/freq_rank_models
python3 plot_freq_rank_models.py [mimic3|mimic4]       # 全范围图，默认 mimic4
python3 plot_freq_rank_models_tail.py [mimic3|mimic4]   # tail 专用图，默认 mimic4
```

依赖：MIMIC-IV train；HALO/LSTM/GPT/AdaPCLA 合成 pkl（seed1）
