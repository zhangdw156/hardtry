# exps/commons 使用说明

实验公共层：**稳定工具**（直接调用）+ **模板**（复制后自包含），互不混用。

## 目录结构

| 路径 | 用途 |
|------|------|
| **bin/** | 稳定工具脚本，供各实验或根目录包装**直接调用**。接口保持稳定，便于维护。 |
| **templates/** | 新建实验时**复制用**的模板，复制到 `exps/<实验名>` 后该实验目录自包含，不依赖 commons 后续改动。 |
| **configs/** | commons 自用配置（convert、default 资源、vllm/eval 示例），也可作为复制参考。 |
| 根目录 `eval_local.sh`、`merge_verl_fsdp_auto.sh` | **兼容包装**，仅转调 `bin/` 内同名脚本，保留旧用法。 |
| **run_example.sh** | 示例：依次执行两段数据转换（hardgen→messages→verl），使用 commons 的 convert 配置。 |

## 实验各步骤用卡数

- **训练（verl）**：由实验目录下 `verl_common_config.yaml`（及若存在的 `verl_common_config_egpo.yaml`）中的 `actor_rollout_ref.num_workers`、`trainer.n_gpus_per_node`、`trainer.nnodes` 控制；脚本会固定 `nnodes: 1`，仅改前两者为「训练用卡数」。
- **评估（vLLM）**：由实验目录下 vLLM 配置（`vllm_config.yaml` 或兼容旧实验的 `vllm_config4.yaml`）中的 `tensor_parallel_size` 控制，卡数可由 **set_exp_gpus.sh** 统一修改。
- 默认值（未传参时）来自 **configs/default_exp_resources.yaml**（`train_n_gpus`、`eval_tensor_parallel_size`），可一次改齐默认用卡再批量跑脚本。

---

## bin/ 稳定工具

### set_exp_gpus.sh

统一设置某实验的「训练用卡数」与「评估用卡数」。

- **用法**：`bash exps/commons/bin/set_exp_gpus.sh <实验目录> [训练用卡数] [评估用卡数]`
- **示例**：`bash exps/commons/bin/set_exp_gpus.sh exps/verl7`（使用默认）、`bash exps/commons/bin/set_exp_gpus.sh exps/verl7 4 4`
- 若省略后两个参数，从 `exps/commons/configs/default_exp_resources.yaml` 读取默认值。
- 会改写该实验目录下 **configs/** 与 **conf/**（若存在）中的：
  - `verl_common_config.yaml`、`verl_common_config_egpo.yaml`：`num_workers`、`n_gpus_per_node`、`nnodes`（nnodes 固定为 1）；
  - `vllm_config.yaml`、`vllm_config4.yaml`（兼容）：`tensor_parallel_size`。
- **适用范围**：verl 实验会同时改训练与评估；swift/full/lora 等仅有 vLLM 配置的会只改评估用卡。

### eval_local.sh

从指定实验目录读取配置，启动 vLLM 并跑评估。

- **用法**：`bash exps/commons/bin/eval_local.sh <实验目录>`
- **示例**：`bash exps/commons/bin/eval_local.sh exps/verl7`
- 会 `cd` 到实验目录，优先使用 `configs/vllm_config.yaml`，不存在则使用 `configs/vllm_config4.yaml`；eval 配置为 `configs/eval_config5.yaml`。需事先在实验目录下准备好对应配置文件。
- 各实验的 `scripts/eval_local.sh` 通常转调本脚本，传入实验目录绝对路径。

### merge_verl_fsdp_auto.sh

在 checkpoint 根目录下取最后一次 `global_step_*` 做 merge。

- **用法**：`bash exps/commons/bin/merge_verl_fsdp_auto.sh <CHECKPOINT_BASE> <TARGET_DIR>`
- **示例**：`bash exps/commons/bin/merge_verl_fsdp_auto.sh /path/to/checkpoints/verl7 /path/to/models/hardtry-4b-verl7`
- 各实验的 `scripts/merge_verl_fsdp_local.sh` 可转调本脚本，并传入该实验的 checkpoint 与 target 路径。

### new_exp.sh

从模板复制生成新实验目录，并替换占位符 `__EXP_NAME__`。

- **用法**：`bash exps/commons/bin/new_exp.sh <verl|swift> <实验名>`
- **示例**：`bash exps/commons/bin/new_exp.sh verl verl9`、`bash exps/commons/bin/new_exp.sh swift full9`
- **verl**：复制 `templates/verl` 到 `exps/<实验名>`，复制后需在 `configs/` 补齐/修改配置（参考 `exps/verl6/configs`），再执行 `bash exps/<实验名>/run_local.sh`。
- **swift**：复制 `templates/swift` 到 `exps/<实验名>`，复制后需补齐 `sft_config.yaml`、`vllm_config.yaml`、`eval_config5.yaml`（参考 `exps/full5/configs`），并修改 `scripts/merge_swift_fsdp_local.sh` 中的 `CKPT_PATH`。

---

## 根目录兼容包装与示例

- **eval_local.sh**：无参时使用 commons 自身目录作为实验目录；有参时转发给 `bin/eval_local.sh`。  
  示例：`bash exps/commons/eval_local.sh`、`bash exps/commons/eval_local.sh exps/verl7`
- **merge_verl_fsdp_auto.sh**：原样转发参数给 `bin/merge_verl_fsdp_auto.sh`。
- **run_example.sh**：依次执行两段 convert，使用 commons 的配置。  
  用法：`bash exps/commons/run_example.sh`（需在仓库根目录或能访问到 `exps/commons/configs` 的路径下执行；内部会 `cd` 到仓库根）。  
  - 步骤 1：`hardtry.utils.convert_hardgen_to_messages`，配置 `exps/commons/configs/convert_hardgen_to_messages_config.yaml`  
  - 步骤 2：`hardtry.utils.convert_messages_to_verl`，配置 `exps/commons/configs/convert_messages_to_verl_config.yaml`

---

## templates/ 模板

- **templates/verl/**：Verl（GRPO）实验骨架，含 `configs/`、`scripts/`、`run_local.sh`。复制后需在 `configs/` 中补齐 `grpo_config.yaml`、`verl_common_config.yaml`、`convert_messages_to_verl_config.yaml`、`vllm_config.yaml`、`eval_config5.yaml` 等（参考 `exps/verl6/configs`）；新实验推荐用 `vllm_config.yaml`（卡数由 set_exp_gpus.sh 管理）。
- **templates/swift/**：ms-swift SFT 实验骨架。复制后需补齐 `sft_config.yaml`、`vllm_config.yaml`、`eval_config5.yaml`（参考 `exps/full5/configs`），并修改 `scripts/merge_swift_fsdp_local.sh` 中的 `CKPT_PATH`。

## 约定

- 各实验的配置统一放在该实验目录下的 **configs/**（部分旧实验使用 **conf/**）。
- 步骤脚本（train/merge/eval）建议放在该实验的 **scripts/**，其中可调用 `exps/commons/bin/` 下的工具。
- 已完成的实验不依赖 commons 的后续修改；新实验可用 `new_exp.sh` 或从 templates 复制生成。
