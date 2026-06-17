# Zero-Shot Code Space 检查结果

## 结论：**生成用的 code 空间与评估用的 code 空间不是同一套**

| 项目 | MIMIC-III (HALO/save) | MIMIC-IV (data2) |
|------|------------------------|------------------|
| **code_vocab_size** | 6984 | 9143 |
| **idToLabel (25 维)** | 25 | 25（一致） |

- **同一 index 对应同一 code 字符串**：仅 **2 / 6984** 个 index 在 III 与 IV 中一致（例如 III 的 index 0 与 IV 的 index 0 几乎总不是同一个 code）。
- **示例**：III index 0 → code `53170`；IV index 0 → code `99973`。III index 100 → `V1869`；IV index 100 → `30182`。
- **code 集合**：有 6439 个 code 字符串在 III 和 IV 中都出现，但在两个数据集中往往对应**不同的 index**（同一诊断在 III 可能是 index 5，在 IV 可能是 index 200）。

## 对 zero-shot 实验的影响

- **model3→IV**：生成时用 **model3（III 的 config）**，输出的 visit 里 code 是 **III 的 index（0..6983）**。评估时 `BASE_DATA_DIR=DATA_IV`，evaluate 脚本用 **IV 的 codeToIndex**（0..9142），把合成数据里的每个 index 当成 **IV 的 code** 来用。
- 因此：**同一个 index 在生成时是 III 的语义，在评估时被当成 IV 的语义**，两边不对齐，下游分类器等于在“错误语义”上训练，在真实 IV 上测试，效果差（如 MyModel2 F1=0）是预期内的。

## 建议（头脑风暴）

1. **不做跨 III/IV 的零样本评估**：若无法对齐两边的 code 空间（例如建立 III index ↔ IV index 或 III code ↔ IV code 的映射），则当前「model3→IV 在 IV 上评估」的设定本身就不合理。
2. **做 code 映射**：若有 III code ↔ IV code 的对应表，可在保存生成数据或送入 evaluate 前，把 III index 转成 IV index（或只保留两边共有的 code 并映射到 IV 的 index），再跑评估。
3. **同域零样本**：例如只在 MIMIC-III 内做「不同站点/时间段」的 π_target，或只在 MIMIC-IV 内做，保证生成与评估使用同一套 code 空间。

（本文件由脚本检查生成，供 Table 3 与零样本实验解读参考。）
