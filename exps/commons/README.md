# exps/commons

实验公共层：**稳定工具**（直接调用）+ **模板**（复制后自包含），互不混用。

## 目录说明

| 路径 | 用途 |
|------|------|
| **bin/** | 稳定工具脚本，供各实验或根目录包装**直接调用**。接口保持稳定，便于维护。 |
| **templates/** | 新建实验时**复制用**的模板，复制到 `exps/<实验名>` 后该实验目录自包含，不依赖 commons 后续改动。 |
| **configs/** | commons 自用的配置与示例（如 `run_example.yaml`），也可作为复制参考。 |
| 根目录 `eval_local.sh`、`merge_verl_fsdp_auto.sh` | **兼容包装**，仅转调 `bin/` 内同名脚本，保留旧用法。 |

## bin/ 稳定工具

- **bin/eval_local.sh**  
  用法：`bash exps/commons/bin/eval_local.sh <实验目录>`  
  从该实验目录的 `configs/` 读取 vLLM 与 eval 配置，启动 vLLM 并跑评估。

- **bin/merge_verl_fsdp_auto.sh**  
  用法：`bash exps/commons/bin/merge_verl_fsdp_auto.sh <CHECKPOINT_BASE> <TARGET_DIR>`  
  在 checkpoint 根目录下取最后一次 `global_step_*` 做 merge。

- **bin/new_exp.sh**  
  用法：`bash exps/commons/bin/new_exp.sh <verl|swift> <实验名>`  
  从对应模板复制到 `exps/<实验名>` 并替换占位符。  
  - `verl`：GRPO/强化学习，复制后参考 `exps/verl6/configs` 补齐配置。  
  - `swift`：ms-swift SFT，复制后参考 `exps/full5/configs` 补齐配置，并修改 `scripts/merge_swift_fsdp_local.sh` 中的 `CKPT_PATH`。

## templates/ 模板

- **templates/verl/**  
  Verl（GRPO）实验骨架：`configs/`、`scripts/`、`run_local.sh`。  
  复制后需在 `configs/` 中补齐 `grpo_config.yaml`、`verl_common_config.yaml`、`convert_messages_to_verl_config.yaml`、`vllm_config4.yaml`、`eval_config5.yaml` 等（参考 `exps/verl6/configs`）。

- **templates/swift/**  
  ms-swift SFT 实验骨架：`configs/`、`scripts/`、`run_local.sh`。  
  复制后需在 `configs/` 中补齐 `sft_config.yaml`、`vllm_config4.yaml`、`eval_config5.yaml`（参考 `exps/full5/configs`），并修改 `scripts/merge_swift_fsdp_local.sh` 中的 `CKPT_PATH` 为本次训练的 checkpoint 路径。

## 约定

- 各实验的配置统一放在该实验目录下的 **configs/**。
- 步骤脚本（train/merge/eval）建议放在该实验的 **scripts/**，其中可调用 `exps/commons/bin/` 下的工具。
- 已完成的实验不依赖 commons 的后续修改；新实验可用 `new_exp.sh` 或从 templates 复制生成。
