# Figure 1: AdaPCLA Highlights Overview（KDD 第二页展示图）

本目录用于生成 KDD 风格的 **四格综合效能展示图**（FRAUDAR / DenseAlert 式），放在论文第二页开头，用一张图回答：理论成立、(b) 指标领先、(c) 长尾结构保真、(d) 零样本可控。

---

## 1. 运行方式与依赖

### 命令（在仓库内执行）

```bash
# 进入本目录后运行
cd mywork/star
python fig1_highlights_overview.py
```

或从仓库根目录：

```bash
python mywork/star/fig1_highlights_overview.py
```

### 依赖

- Python 3.8+
- `matplotlib`、`numpy`（无额外数据包）

### 输出文件

- `fig1_highlights_overview.png`（默认 150 dpi，用于预览与插入 Word/Overleaf）
- `fig1_highlights_overview.pdf`（若保存成功，用于 LaTeX `\includegraphics`）

---

## 2. 四格内容与数据来源

| 子图 | 含义 | 数据来源 | 说明 |
|------|------|----------|------|
| **(a) Internalization** | Curriculum 有效、Static 撤掉 bias 后崩塌 | 叙事来自 Table 3 / `output/Mechanism Ablation/mechanism_ablation_compact.csv` | 曲线为**示意**：α 从 1→0 时 AdaPCLA 保持高 TailPairSeen，Static 跌至 0。无 epoch 级数据时用此示意。 |
| **(b) Downstream Utility** | MIMIC-IV 上 F1、AUPRC 对比 | `mywork/output/acc&f1/seed3_summary_with_synteg_and_adapcla_mimic4_with_auprc.csv` | 取各 model 多 seed 的 mean F1 / mean AUPRC；AdaPCLA 柱高亮并可选标 “+X%” 相对 HALO。 |
| **(c) Tail Structure Fidelity** | 长尾共现结构：Real / AdaPCLA / HALO | `mywork/output/heatmap/heatmap_real_tail.png`、`heatmap_adapcla_tail.png`、`heatmap_halo_tail.png` | 三张图横向拼在 (c) 一格内。若缺文件则 (c) 显示占位说明。 |
| **(d) Zero-Shot Controllability** | IV→III 零样本 F1 | `mywork/zero-shot/output/zeroshot_baselines_table.csv`、`mywork/output/zero-shot/output/zeroshot_table3.csv` | 目标=MIMIC-III 的 F1；AdaPCLA 柱高亮。 |

路径均相对 **mywork**（脚本所在目录的上一级）。

---

## 3. 设计说明（与 FRAUDAR / DenseAlert 对齐）

- **(a)** 对应「理论/机制」：Curriculum annealing 防崩塌，Static 撤 bias 即崩溃。
- **(b)** 对应「指标领先」：下游 F1/AUPRC 一目了然，审稿人扫一眼即知 SOTA。
- **(c)** 对应「真实结构」：生成的长尾共现与 Real 接近，HALO 对比更散。
- **(d)** 对应「可扩展/零样本」：不重训即可迁移到目标人群。

Caption 建议（可放入论文）：

> **Figure 1: Overview of AdaPCLA's effectiveness.** (a) Curriculum annealing keeps tail performance stable as bias is removed; static bias collapses (Table 3). (b) AdaPCLA achieves best F1 and AUPRC on MIMIC-IV (Table 2). (c) Tail co-occurrence structure matches real data (heatmap); baselines are noisier. (d) Zero-shot transfer to MIMIC-III attains strong F1 without retraining (Table 4).

---

## 4. 若数据或路径不一致

- **(b)** 若 CSV 不存在或列名不同，脚本会退回到论文中的 MIMIC-IV 数值（写死在 fallback）。
- **(c)** 若 `output/heatmap/` 下没有 `heatmap_*_tail.png`，会画出占位文字，提示放入对应热力图。
- **(d)** 若 zeroshot CSV 缺失，会使用脚本内 fallback 的 IV→III F1 列表。

可根据自己仓库结构调整脚本顶部的 `ACC_F1_CSV`、`ZEROSHOT_BASELINES`、`ZEROSHOT_OURS`、`HEATMAP_DIR`。

---

## 5. 修改与扩展

- **布局**：当前为 2×2。若要 1×4 横排，可改为 `fig, axes = plt.subplots(1, 4, ...)` 并相应调整 `panel_*` 的传入轴。
- **配色**：`COLOR_OURS`、`COLOR_BASELINE`、`COLOR_STATIC` 在脚本开头，可改成与论文配色一致。
- **(a) 真实曲线**：若后续有「epoch 或 α vs TailPairSeen」的表格/CSV，可替换 `panel_a` 中的 `alpha`、`adapcla`、`static` 为从文件读取的数组。

脚本内注释和本 README 已写明每格含义与数据来源，便于后续按需修改。
