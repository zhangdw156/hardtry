# Verl 实验模板

用 `exps/commons/bin/new_exp.sh verl <实验名>` 从本模板复制到 `exps/<实验名>`。

## 配置约定

每个 verl 实验**主配置为三个文件**，依赖关系：**grpo_config → verl_train_config → verl_config**。

| 文件 | 用途 |
|------|------|
| **verl_config.yaml** | 实验元信息：实验名、工作根路径、模型根路径、各 venv、checkpoint/评估/数据等路径（被 verl_train_config 引用）。 |
| **verl_train_config.yaml** | 训练配置：算法、数据、actor/rollout、trainer 超参、reward、Ray 等；路径通过 `${verl_config.xxx}` 引用；由 grpo_config 引入。 |
| **grpo_config.yaml** | 入口配置：train_local.sh 使用 `--config-name=grpo_config`；指定本实验 experiment_name、searchpath，并引入 verl_train_config。 |

- **vllm_config.yaml**、**eval_config5.yaml**、**convert_messages_to_verl_config.yaml** 为各流水线所需，路径等元信息以 `verl_config.yaml` 为准，修改元信息时请与之同步。

## 占位符（均在 verl_config 中；new_exp.sh 只替换 __EXP_NAME__）

| 占位符 | 说明 | 生成时 |
|--------|------|--------|
| `__EXP_NAME__` | 实验名 | 自动替换 |
| `__WORK_ROOT__` | 工作/仓库根路径 | 需手动替换 |
| `__MODELS_ROOT__` | 模型根路径 | 需手动替换 |
| `__VENV_VERL__` | verl 虚拟环境路径 | 需手动替换 |
| `__VENV_GORILLA__` | 评估用 venv 路径 | 需手动替换 |

生成后在实验目录做一次全局替换上述占位符；实验参数（batch、GPU 数等）在 **verl_train_config.yaml** 中按注释修改。执行：

`bash exps/<实验名>/run_local.sh`
