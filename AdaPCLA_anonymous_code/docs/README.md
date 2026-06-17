# Zero-Shot Controllability (Baselines + AdaPCLA)

本目录用于跑 **zero-shot** 对比：在源域训练、目标域不做训练，直接做 25-label 下游评估。

- **AdaPCLA**：由 `../output/zero-shot/run_zeroshot.sh` 产出，结果在 `../output/zero-shot/output/zeroshot_table3.csv`。
- **Baselines（GPT / LSTM / EVA / SynTEG / HALO）**：由本目录脚本产出，结果在 `output/zeroshot_baselines_table.csv`。

## 环境与依赖

- 激活你的 Python 环境（如 `conda activate sft_lab`）。
- 需安装 PyTorch，且能运行 `torchrun`（多卡时设置 `NUM_GPUS`）。
- 数据与模型路径见 `paths_baselines.py`（默认：`fame/myfame/baseline/` 下各 baseline 的 `save/`、`save_mimiciv_seed1/`）。

## 可执行脚本

在 **`mywork/zero-shot`** 下执行（建议先 `chmod +x *.sh`）：

| 脚本 | 说明 |
|------|------|
| `./run_all_zeroshot.sh` | 先跑 AdaPCLA zero-shot，再跑 5 个 baseline 的 zero-shot（III→IV、IV→III） |
| `./run_gpt_zeroshot.sh` | 仅 GPT zero-shot |
| `./run_lstm_zeroshot.sh` | 仅 LSTM zero-shot |
| `./run_eva_zeroshot.sh` | 仅 EVA zero-shot |
| `./run_synteg_zeroshot.sh` | 仅 SynTEG zero-shot |
| `./run_halo_zeroshot.sh` | 仅 HALO zero-shot |

单卡时默认 `NUM_GPUS=1`；多卡可先设置再跑，例如：

```bash
export NUM_GPUS=4
./run_all_zeroshot.sh
```

只跑某几个 baseline 或某方向时，可直接用 Python：

```bash
python3 run_baselines_zeroshot.py --baselines gpt,halo --directions iii_to_iv
```

## 输出

- **Baselines**：`output/zeroshot_baselines_table.csv`（列：target, method, Acc, F1, AUPRC）。
- **AdaPCLA**：`../output/zero-shot/output/zeroshot_table3.csv`。
- 各 baseline 的映射后生成数据：`output/{gpt,lstm,eva,synteg,halo}_iii_to_iv_mapped.pkl`、`*_iv_to_iii_mapped.pkl`。
- 下游评估明细：`output/eval_{baseline}_to_iv/`、`output/eval_{baseline}_to_iii/` 下的 `compare_real_halo_mymodel2.csv`。
