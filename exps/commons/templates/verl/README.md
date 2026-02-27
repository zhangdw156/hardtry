# Verl 实验模板

新建 verl 类实验时，用 `exps/commons/bin/new_exp.sh verl <实验名>` 从本模板复制到 `exps/<实验名>`。

复制后需要你手动完成：

1. **configs/** 下补齐/修改配置：参考 `exps/verl6/configs/`，至少需要  
   `grpo_config.yaml`、`verl_common_config.yaml`、`convert_messages_to_verl_config.yaml`、  
   `vllm_config4.yaml`、`eval_config5.yaml`（其中 `experiment_name`、`summary_output_dir`、  
   vLLM 的 `model`、merge 的 checkpoint/target 等按实验名修改）。
2. **scripts/merge_verl_fsdp_local.sh** 中的 `CHECKPOINT_BASE`、`TARGET_DIR` 已按实验名生成，若路径规范不同请自行修改。

入口：在仓库根目录执行 `bash exps/<实验名>/run_local.sh`，或  
`uv run python -m hardtry.run config_file=exps/<实验名>/configs/run_<实验名>.yaml`。
