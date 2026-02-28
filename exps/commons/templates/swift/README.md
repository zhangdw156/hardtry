# ms-swift 实验模板

新建 ms-swift（SFT）实验时，用 `exps/commons/bin/new_exp.sh swift <实验名>` 从本模板复制到 `exps/<实验名>`。

复制后需要你手动完成：

configs 已内置（sft_config、vllm_config、eval_config5 等），占位符与 verl 模板一致（`__EXP_NAME__` 自动替换，`__WORK_ROOT__`、`__MODELS_ROOT__`、`__VENV_*` 需在实验目录中手动替换一次）。merge 脚本中的 `CKPT_PATH`、`OUTPUT_PATH` 已用相同占位符，与 configs 同步替换即可。

执行：`bash exps/<实验名>/run_local.sh`
