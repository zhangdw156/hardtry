# baseline_4bthinking

**仅用于验证**：Qwen3-4B-Thinking-2507 在 BFCL multi-turn base 上的表现，无任何训练。

- 模型：`/dfs/data/models/Qwen3-4B-Thinking-2507`（原始 Thinking，无 LoRA）
- 评估：`test_category: multi_turn_base`，5 轮并行，结果写入 `eval5_results/`

## 运行评估

```bash
bash exps/baseline_4bthinking/scripts/eval_local.sh
```

或从仓库根目录：

```bash
bash exps/commons/bin/eval_local.sh exps/baseline_4bthinking
```
