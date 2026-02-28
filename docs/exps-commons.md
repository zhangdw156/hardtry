# exps/commons 使用说明

实验公共层：**bin**（创建新实验、改卡数等，用户直接调用）、**sbin**（仅由实验内 scripts 转调）、**configs**（本机全局配置）、**templates**（verl/swift 模板）。

## 目录结构

| 路径 | 用途 |
|------|------|
| **bin/** | 用户直接调用的工具：新建实验、设置用卡数等。 |
| **sbin/** | 仅由实验目录下 `scripts/*.sh` 转调，不对外单独使用（eval_local、merge_verl_fsdp_auto）。 |
| **configs/** | 本机全局配置，一台机器配置一次即可，基本不随实验改动。 |
| **templates/** | verl / swift 实验模板，`new_exp.sh` 复制到 `exps/<实验名>` 后自包含。 |

## configs/ 全局配置

**configs/global.yaml** 为本机唯一全局配置，建议克隆仓库后按本机环境填写一次（勿提交个人路径）：

- **路径**：`work_root`、`models_root`、`venv_verl`、`venv_gorilla`、`venv_swift`、`venv_merge`
- **默认用卡**：`train_n_gpus`、`eval_tensor_parallel_size`（供 set_exp_gpus.sh 未传参时使用）

**new_exp.sh** 会据此替换模板中的 `__WORK_ROOT__`、`__MODELS_ROOT__`、`__VENV_*` 等占位符；若未配置 global.yaml，生成后的实验目录内需手动替换这些占位符。

---

## bin/ 用户工具

### new_exp.sh

从模板复制生成新实验目录，并替换 `__EXP_NAME__`；若存在 **configs/global.yaml** 则同时替换全局路径占位符。

- **用法**：`bash exps/commons/bin/new_exp.sh <verl|swift> <实验名>`
- **示例**：`bash exps/commons/bin/new_exp.sh verl verl9`、`bash exps/commons/bin/new_exp.sh swift full9`
- **verl**：复制 `templates/verl`，configs 已内置，按 global 替换后即可执行 `bash exps/<实验名>/run_local.sh`。
- **swift**：同上，复制 `templates/swift`。

### apply_global_config.sh

对已有实验目录应用 **configs/global.yaml**，将 `__WORK_ROOT__`、`__MODELS_ROOT__`、`__VENV_*` 等占位符替换为本机路径。适用于非 new_exp 生成的实验（如 baseline_8b），或修改过 global.yaml 后希望同步的实验。

- **用法**：`bash exps/commons/bin/apply_global_config.sh <实验目录>`
- **示例**：`bash exps/commons/bin/apply_global_config.sh exps/baseline_8b`

### set_exp_gpus.sh

统一设置某实验的「训练用卡数」与「评估用卡数」。

- **用法**：`bash exps/commons/bin/set_exp_gpus.sh <实验目录> [训练用卡数] [评估用卡数]`
- **示例**：`bash exps/commons/bin/set_exp_gpus.sh exps/verl7`（使用默认）、`bash exps/commons/bin/set_exp_gpus.sh exps/verl7 4 4`
- 若省略后两个参数，从 **exps/commons/configs/global.yaml** 读取 `train_n_gpus`、`eval_tensor_parallel_size`。
- 会改写：**configs/** 与 **conf/** 下的 `verl_train_config.yaml`、`verl_common_config*.yaml`（训练卡数）、`vllm_config*.yaml`（评估 tensor_parallel_size）；若存在 **scripts/train_local.sh** 且含 `NPROC_PER_NODE`/`CUDA_VISIBLE_DEVICES`（Swift 实验），则一并按训练用卡数更新。

---

## sbin/ 实验脚本转调

以下脚本仅供实验目录内 `scripts/*.sh` 转调使用，不建议直接对外调用。

### eval_local.sh

在指定实验目录下启动 vLLM，再执行 BFCL 评估。

- **用法**：`bash exps/commons/sbin/eval_local.sh <实验目录>`
- 实验目录须包含 `configs/vllm_config.yaml` 与 `configs/eval_config5.yaml`。
- 各实验的 `scripts/eval_local.sh` 转调本脚本并传入实验目录绝对路径。

### merge_verl_fsdp_auto.sh

将 checkpoint 根目录下**最后一次** `global_step_*` 的 actor 合并到目标目录。

- **用法**：`bash exps/commons/sbin/merge_verl_fsdp_auto.sh <CHECKPOINT_BASE> <TARGET_DIR>`
- 各实验的 `scripts/merge_verl_fsdp_local.sh` 转调本脚本并传入路径。
- 可选环境变量：`PYTHON_CMD`（默认 `/dfs/data/uv-venv/verl/bin/python3`）。

### merge_swift_fsdp_auto.sh

将 Swift/FSDP checkpoint 根目录下**最近一次** run（v*-*）与最大 checkpoint-N 合并为完整模型，并拷贝 tokenizer 等。

- **用法**：`bash exps/commons/sbin/merge_swift_fsdp_auto.sh <CHECKPOINT_BASE> <TARGET_DIR> [BASE_MODEL_PATH]`
- 各实验的 `scripts/merge_swift_fsdp_local.sh` 转调本脚本并传入路径；可选第三参为拷贝 tokenizer 用的 base 模型目录。
- 需在已 activate 的 merge 用 venv（如 huggingface）下执行。

---

## templates/ 模板

- **templates/verl/**：Verl（GRPO）实验骨架，configs 已内置（verl_config、verl_train_config、grpo_config、vllm、eval、convert 等）；依赖关系 grpo_config → verl_train_config → verl_config；占位符由 new_exp + global 替换，卡数由 set_exp_gpus.sh 管理。
- **templates/swift/**：ms-swift SFT 实验骨架，configs 已内置，占位符同上。

## 约定

- 各实验的配置统一放在该实验目录下的 **configs/**（部分旧实验使用 **conf/**）。
- 步骤脚本（train/merge/eval）放在该实验的 **scripts/**，其中转调 `exps/commons/sbin/` 下的脚本。
- 本机全局信息（路径、默认用卡）只维护 **exps/commons/configs/global.yaml** 一处；新实验生成时自动带入，无需在实验内重复修改。
