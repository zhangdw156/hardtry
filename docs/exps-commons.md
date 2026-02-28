# exps/commons 使用说明

实验公共层：**稳定工具**（直接调用）+ **模板**（复制后自包含），互不混用。

## 目录结构

| 路径 | 用途 |
|------|------|
| **bin/** | 稳定工具脚本，**唯一入口**。各实验通过 `scripts/` 转调或直接从仓库根调用。 |
| **templates/** | 新建实验时复制用模板，复制到 `exps/<实验名>` 后自包含，configs 已内置。 |
| **default_exp_resources.yaml** | set_exp_gpus.sh 未传参时的默认用卡数。 |

## 实验各步骤用卡数

- **训练（verl）**：由实验目录下 `verl_common_config.yaml`（及若存在的 `verl_common_config_egpo.yaml`）中的 `actor_rollout_ref.num_workers`、`trainer.n_gpus_per_node`、`trainer.nnodes` 控制；脚本会固定 `nnodes: 1`，仅改前两者为「训练用卡数」。
- **评估（vLLM）**：由实验目录下 vLLM 配置（`vllm_config.yaml` 或兼容旧实验的 `vllm_config4.yaml`）中的 `tensor_parallel_size` 控制，卡数可由 **set_exp_gpus.sh** 统一修改。
- 默认值（未传参时）来自 **exps/commons/default_exp_resources.yaml**（`train_n_gpus`、`eval_tensor_parallel_size`），可一次改齐默认用卡再批量跑脚本。

---

## bin/ 稳定工具

### set_exp_gpus.sh

统一设置某实验的「训练用卡数」与「评估用卡数」。

- **用法**：`bash exps/commons/bin/set_exp_gpus.sh <实验目录> [训练用卡数] [评估用卡数]`
- **示例**：`bash exps/commons/bin/set_exp_gpus.sh exps/verl7`（使用默认）、`bash exps/commons/bin/set_exp_gpus.sh exps/verl7 4 4`
- 若省略后两个参数，从 `exps/commons/default_exp_resources.yaml` 读取默认值。
- 会改写该实验目录下 **configs/** 与 **conf/**（若存在）中的：
  - `verl_common_config.yaml`、`verl_common_config_egpo.yaml`：`num_workers`、`n_gpus_per_node`、`nnodes`（nnodes 固定为 1）；
  - `vllm_config.yaml`、`vllm_config4.yaml`（兼容）：`tensor_parallel_size`。
- **适用范围**：verl 实验会同时改训练与评估；swift/full/lora 等仅有 vLLM 配置的会只改评估用卡。

### eval_local.sh

在指定实验目录下启动 vLLM，再执行 BFCL 评估。

- **用法**：`bash exps/commons/bin/eval_local.sh <实验目录>`
- **示例**：`bash exps/commons/bin/eval_local.sh exps/verl7`
- 实验目录须包含 `configs/vllm_config.yaml` 与 `configs/eval_config5.yaml`。vLLM 就绪后调用 `hardtry.utils.eval_runner`。
- 各实验的 `scripts/eval_local.sh` 转调本脚本并传入实验目录绝对路径。

### merge_verl_fsdp_auto.sh

将 checkpoint 根目录下**最后一次** `global_step_*` 的 actor 合并到目标目录。

- **用法**：`bash exps/commons/bin/merge_verl_fsdp_auto.sh <CHECKPOINT_BASE> <TARGET_DIR>`
- **示例**：`bash exps/commons/bin/merge_verl_fsdp_auto.sh /path/to/checkpoints/verl7 /path/to/models/hardtry-4b-verl7`
- 可选环境变量：`PYTHON_CMD`（默认 `/dfs/data/uv-venv/verl/bin/python3`）。各实验的 `scripts/merge_verl_fsdp_local.sh` 转调本脚本并传入路径。

### new_exp.sh

从模板复制生成新实验目录，并替换占位符 `__EXP_NAME__`。

- **用法**：`bash exps/commons/bin/new_exp.sh <verl|swift> <实验名>`
- **示例**：`bash exps/commons/bin/new_exp.sh verl verl9`、`bash exps/commons/bin/new_exp.sh swift full9`
- **verl**：复制 `templates/verl` 到 `exps/<实验名>`，configs 已内置，按需替换占位符后执行 `bash exps/<实验名>/run_local.sh`。
- **swift**：复制 `templates/swift` 到 `exps/<实验名>`，configs 已内置，按需替换占位符后执行 `bash exps/<实验名>/run_local.sh`。

两段数据转换（hardgen→messages→verl）需在实验目录下执行，使用该实验 `configs/` 中的配置（如从 templates/verl 生成后的 `convert_messages_to_verl_config.yaml` 等），或自行准备配置文件路径。

---

## templates/ 模板

- **templates/verl/**：Verl（GRPO）实验骨架，configs 已内置（verl_config、verl_common_config、vllm、eval、convert 等），占位符替换后即可运行；卡数由 set_exp_gpus.sh 管理。
- **templates/swift/**：ms-swift SFT 实验骨架，configs 已内置，占位符替换后即可运行。

## 约定

- 各实验的配置统一放在该实验目录下的 **configs/**（部分旧实验使用 **conf/**）。
- 步骤脚本（train/merge/eval）建议放在该实验的 **scripts/**，其中可调用 `exps/commons/bin/` 下的工具。
- 已完成的实验不依赖 commons 的后续修改；新实验可用 `new_exp.sh` 或从 templates 复制生成。
