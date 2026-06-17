# Real Case Study (MIMIC-IV)

用于论文 Qualitative Analysis 的 Case A（单条轨迹对比）与 Case B（tail 共现）结果。

## 数据与模型

- **Real**：MIMIC-IV `data2` 的 test（Case A）与 train（Case B）。
- **HALO / LSTM / GPT / AdaPCLA**：MIMIC-IV 上训练的模型在 `data2` 上的**已生成**合成数据（与 Table 1 一致）。LSTM、GPT 路径见脚本内 `LSTM_SYN_PKL`、`GPT_SYN_PKL`。

## 输出文件

| 文件 | 说明 |
|------|------|
| `run_case_study.py` | 主脚本：Case A 多 baseline（HALO/LSTM/GPT/AdaPCLA）轨迹对比；Case B 在约 30 个 tail 中选 Real–AdaPCLA 重叠最高的 tail 并输出双列表格。 |
| `case_a_result.json` | Case A 的原始结果：Real、各 baseline、AdaPCLA 的 visit1/2/3 的 code 列表。 |
| `case_a_table.csv` | Case A 表格：行 Real / HALO / LSTM / GPT / AdaPCLA，列 Visit 1 / 2 / 3。 |
| `case_b_result.json` | Case B 的原始结果：多 tail 的 Real/HALO/AdaPCLA Top-K 共现，及 `best_tail_id`、`best_overlap`。 |
| `case_b_best_tail.json` | Case B 展示用 tail 信息：best_tail_id, code, tail_name, best_overlap。 |
| `case_b_table_left.csv` | Case B 论文用左表：Rank, Real, AdaPCLA（Top-5，best tail）。 |
| `case_b_table_right.csv` | Case B 论文用右表：Rank, Real, HALO（Top-5，best tail）。 |
| `case_b_tail_*_cooccur.csv` | Case B 各候选 tail 的完整共现表（Source, Rank, Co-occurring code, Name, Count）。 |

## Case A 逻辑

1. 在 test 集中筛出「至少 3 次 visit 且最后一次 visit 含至少一个 tail code」的真实患者候选。
2. **AdaPCLA**：对每个真实患者，在合成数据中先按「前两 visit 与 Real 的 Jaccard 平均」取 context 最相似的若干候选（top 800），再在这些候选中选 **Visit 3 与 Real Visit 3 重叠最高**（Jaccard）的一条。
3. **HALO / LSTM / GPT**：对同一真实患者，仅在各自合成数据中取「前两 visit 与 Real 最相似」的一条（不优化 Visit 3 重叠）。
4. 在所有真实候选中，选取能使 **AdaPCLA Visit 3 与 Real Visit 3 的 Jaccard 最大**的那条真实患者，作为最终案例。表格呈现为：在相同 context 下，AdaPCLA 的 Visit 3 与 ground truth 高度重合，各 baseline 的 Visit 3 则单码或明显更差。

## Case B 逻辑

1. 从 tail bucket 中选约 30 个在 train 里出现至少 2 次的 tail code，对每个在 Real/HALO/AdaPCLA 中统计 Top-K 共现。
2. 计算每个 tail 的 **Real Top-5 与 AdaPCLA Top-5 的交集大小**，选取 **重叠最大** 的 tail 作为论文展示用。
3. 输出双列：左表 Real vs AdaPCLA、右表 Real vs HALO（均为该 best tail 的 Top-5），避免单表列宽过长。

## 论文中使用建议

- **Case A**：将 `case_a_table.csv` 转为 LaTeX 表（或根据需要精简/加粗 tail code），配 1–2 句点评：Real 中 tail 与上下文一致；HALO 的 Visit 3 仅单码或与上下文脱节；AdaPCLA 的 Visit 3 多码且与 Real 更一致。
- **Case B**：引用 `case_b_tail_*_cooccur.csv` 中的 Top 共现，用 1–2 句说明 AdaPCLA 的 tail 共现邻域更接近 Real，与 Table 1 的 TailPairSeen/TailTopKJac 一致。
- 在 caption 中注明为 **illustrative / representative** case，来自 MIMIC-IV test（Case A）与 train（Case B）。

## 重新运行

```bash
cd "EXPERIMENTS_ROOT/output/real case"
python3 run_case_study.py
```

依赖：`fame/myfame/data2` 下 test/train、codeToIndex、indexToCode；`fame/myfame/output/长尾分布问题分析/mimiciv_code_buckets.csv`；HALO 与 AdaPCLA 的合成 pkl；可选 `heatmap/D_ICD_DIAGNOSES.csv` 用于 SHORT_TITLE 名称。
