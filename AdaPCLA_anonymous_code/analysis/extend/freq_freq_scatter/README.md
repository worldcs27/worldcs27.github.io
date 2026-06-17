# 方案 C：Real vs Generated 频率–频率散点图

每个代码（code）为一点：横轴 = 真实数据频率，纵轴 = 生成模型该代码频率。  
一个子图对应一个模型（HALO / LSTM / GPT / AdaPCLA），对角线 y=x 表示与真实频率一致。

## 用法

```bash
python3 plot_freq_freq_scatter.py [mimic3|mimic4]   # 默认 mimic4
```

## 输出

- `output/freq_freq_scatter_mimic4.png`
- `output/freq_freq_scatter_mimic3.png`

坐标轴为 symlog（含 0，低频线性、高频对数），便于同时查看头部与长尾代码。
