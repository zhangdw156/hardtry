# ms-swift 实验模板

新建 ms-swift（SFT）实验时，用 `exps/commons/bin/new_exp.sh swift <实验名>` 从本模板复制到 `exps/<实验名>`。

复制后需要你手动完成：

1. **configs/** 下补齐配置（可复制自 `exps/full5/configs` 或 `exps/commons/configs` 再改）：
   - **sft_config.yaml**：模型、数据路径、训练超参等（必填）。
   - **vllm_config.yaml**：推理用模型路径（merge 后的模型路径，如 `hardtry-4b-<实验名>`）；卡数可由 set_exp_gpus.sh 修改。
   - **eval_config5.yaml**：`experiment_name`、`summary_output_dir` 等按实验名修改。
2. **scripts/merge_swift_fsdp_local.sh**：将 `CKPT_PATH` 改为你本次训练产生的 checkpoint 路径（如 `.../checkpoints/<实验名>/v0-YYYYMMDD-HHMMSS/checkpoint-N/pytorch_model_fsdp_0`）。`OUTPUT_PATH` 已按实验名生成。

入口：`bash exps/<实验名>/run_local.sh`（可从任意目录执行）。
