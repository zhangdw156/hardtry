# ms-swift 实验模板

用 `exps/commons/bin/new_exp.sh swift <实验名>` 从本模板复制到 `exps/<实验名>`。

configs 已内置（sft_config、vllm_config、eval_config5）。占位符与 verl 一致：`__EXP_NAME__` 自动替换；以下需在实验目录中手动替换一次：

| 占位符 | 说明 |
|--------|------|
| `__WORK_ROOT__` | 工作/仓库根路径 |
| `__MODELS_ROOT__` | 模型根路径 |
| `__VENV_SWIFT__` | SFT 训练用 venv（如 modelscope） |
| `__VENV_MERGE__` | merge 用 venv（如 huggingface） |
| `__VENV_GORILLA__` | 评估用 venv（eval_config5） |

merge 步骤会**自动**使用本实验 checkpoint 目录下最近一次 run（v*-*）与最大 checkpoint-N，无需改路径。执行：

`bash exps/<实验名>/run_local.sh`
