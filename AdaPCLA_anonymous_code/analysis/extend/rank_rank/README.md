# 方案 D：Rank–rank 图

每个代码为一点：横轴 = 真实数据中的排名（1 = 最频繁），纵轴 = 生成模型中的排名。  
一个子图对应一个模型（HALO / LSTM / GPT / AdaPCLA），对角线 y=x 表示排名一致。

## 用法

```bash
python3 plot_rank_rank.py [mimic3|mimic4]   # 默认 mimic4
```

## 输出

- `output/rank_rank_mimic4.png`
- `output/rank_rank_mimic3.png`

横纵轴均为对数刻度，便于观察全排名范围。未在生成数据中出现的代码其生成排名设为 max_rank+1。
