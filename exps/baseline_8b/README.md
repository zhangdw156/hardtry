# baseline_8b

**仅用于验证**：Qwen3-8B 在 BFCL multi-turn base 上的表现，无任何训练。

- 模型：`/dfs/data/models/Qwen3-8B`（原始 8B，无 LoRA）
- 评估：`test_category: multi_turn_base`，5 轮并行，结果写入 `eval5_results/`
- 用卡数可由 `bash exps/commons/bin/set_exp_gpus.sh exps/baseline_8b [训练用卡] [评估用卡]` 修改（本实验仅评估，仅会改 vLLM 的 tensor_parallel_size）

## 运行评估

```bash
bash exps/baseline_8b/scripts/eval_local.sh
```

或从仓库根目录：

```bash
bash exps/commons/bin/eval_local.sh exps/baseline_8b
```
