# src/hardtry 使用说明

`hardtry` 是项目 Python 包，提供数据转换、评估、模型合并与奖励函数等能力，供 exps 脚本或 VeRL 配置调用。

## 包结构概览

```
src/hardtry/
├── __init__.py          # 版本号
├── utils/               # 工具模块
│   ├── convert_hardgen_to_messages.py   # BFCL/hardgen 原始数据 → OpenAI messages JSON
│   ├── convert_messages_to_verl.py     # messages JSON → VeRL parquet（含 reward 结构）
│   └── eval_runner.py                   # 并行 BFCL 评估（generate + evaluate + 结果汇总）
└── rl/                  # 强化学习奖励
    ├── reward_fn.py     # 通用 tool_call 奖励（格式 + 正确性分开计分）
    ├── reward_fn_egpo.py # EGPO 用严格二元奖励（<think> 格式 + AST 全对才 1 分）
    └── reward_fn_grpo.py # GRPO 用二元奖励（不要求 think，用于 Instruct 模型如 verl9）
```

---

## utils：数据转换与评估

### convert_hardgen_to_messages

将原始 JSON 数据（如 BFCL multi-turn 格式）转换为 OpenAI messages 格式的 JSON 文件，便于后续转 VeRL 或做 SFT。

- **入口**：`python -m hardtry.utils.convert_hardgen_to_messages [config.yaml]` 或 `--input ... --output ... --max_samples ...`
- **配置（YAML）示例**：
  - `input`：输入 JSON 路径
  - `output`：输出 messages JSON 路径
  - `max_samples`：保留条数，-1 表示全部
- **用法示例**：使用某实验目录下 configs（如从 templates 生成后的 `exps/<实验名>/configs/convert_hardgen_to_messages_config.yaml`）或自行准备配置文件路径。

### convert_messages_to_verl

将 messages 格式 JSON 转为 VeRL 训练用的 parquet（含 prompt、ground_truth、reward_model 等字段）。

- **入口**：`python -m hardtry.utils.convert_messages_to_verl [config_file=]path.yaml` 或通过 Hydra/OmegaConf 传参。
- **配置（YAML）常用字段**：
  - `input_path`：messages JSON 路径
  - `output_dir`：输出目录（会在此目录下按 data_source/ability 等写出 parquet）
  - `model_path`：tokenizer 对应模型路径
  - `test_size`、`seed`、`num_proc`：划分与预处理
  - `data_source`、`ability`：数据来源与能力标签
  - `max_samples`：可选，只保留前 N 条
- **用法示例**：  
  `uv run python -m hardtry.utils.convert_messages_to_verl exps/verl7/configs/convert_messages_to_verl_config.yaml`  
  或由各实验的 `scripts/convert_messages_to_verl.sh` 调用；也可手动指定 commons 下 convert 配置。

### eval_runner

在已启动的 vLLM 服务上，并行跑多轮 BFCL generate + evaluate，并汇总 CSV 结果。

- **入口**：`python -m hardtry.utils.eval_runner <eval_config5.yaml>` 或通过 HfArgumentParser 传参。
- **配置（YAML）常用字段**：
  - `model_name`、`test_category`：BFCL 模型名与测试类别（如 multi_turn_base）
  - `remote_openai_base_url`、`remote_openai_api_key`、`remote_openai_tokenizer_path`：vLLM 服务与本地 tokenizer
  - `num_runs`、`threads_per_run`：并行轮数与每轮线程数
  - `base_artifact_dir`、`experiment_name`：单次实验日志与结果根目录
  - `summary_output_dir`：汇总 CSV 的目标目录（如各实验的 `eval5_results`）
- **用法示例**：  
  先由 `exps/commons/sbin/eval_local.sh <实验目录>` 启动 vLLM，再在脚本内部调用：  
  `uv run -m hardtry.utils.eval_runner "$EVAL_CONFIG_ABS"`（EVAL_CONFIG 一般为该实验的 `configs/eval_config5.yaml`）。

---

## rl：奖励函数

VeRL 训练时通过配置指定自定义 reward 模块，入口函数为 `compute_score(data_source, solution_str, ground_truth, extra_info=None)`，返回浮点分数。

### reward_fn.py（通用 GRPO）

- **用途**：格式与正确性分开计分。含 `<tool_call>...</tool_call>` 给格式分（如 0.1），与 ground_truth 的 tool_call 内容一致再给正确性分（如 1.0），总分可为 0 / 0.1 / 1.0 / 1.1 等。
- **适用**：一般 GRPO 或需要细粒度奖励的实验。
- **入口**：`hardtry.rl.reward_fn.compute_score`（由 VeRL 配置中的 `reward_model` 等引用）。

### reward_fn_egpo.py（EGPO 严格二元）

- **用途**：严格 0/1。先校验格式为 `<think>...</think>...`（思考块 + 后续内容），再仅对 `</think>` 之后的内容做 `<tool_call>` 解析并与 ground_truth 比较；格式正确且 AST 全对才返回 1.0，否则 0.0。
- **适用**：EGPO 及与 EGPO 公平对比的 GRPO（如 verl7/verl8）。
- **入口**：`hardtry.rl.reward_fn_egpo.compute_score`（在实验配置中指定该模块）。

### reward_fn_grpo.py（GRPO 二元，不要求 think）

- **用途**：严格 0/1，不校验 `<think>`。对整段 solution 做 `<tool_call>` 解析并与 ground_truth 比较，一致则 1.0，否则 0.0。适用于 Qwen3-4B-Instruct 等不输出思考块的模型（如 verl9）。
- **入口**：`hardtry.rl.reward_fn_grpo.compute_score`（在实验配置中指定该模块）。

在实验的 `verl_train_config.yaml`（或旧实验的 `verl_common_config.yaml`、EGPO 的 `verl_common_config_egpo.yaml`）中通过 `custom_reward_function.path` 等指向对应模块即可。

---

## 与 exps 的配合

- **数据准备**：两段 convert 可手动按文档执行；各实验的 convert 脚本调用 `convert_messages_to_verl`，配置放在实验目录 `configs/` 下。
- **评估**：`exps/commons/sbin/eval_local.sh` 启动 vLLM 后调用 `hardtry.utils.eval_runner`，eval 配置由实验目录的 `configs/eval_config5.yaml` 提供。
- **奖励**：verl7/verl8 在 VeRL 配置中引用 `hardtry.rl.reward_fn_egpo`；verl9（GRPO + Instruct）引用 `hardtry.rl.reward_fn_grpo`；其它 GRPO 实验可引用 `hardtry.rl.reward_fn`。

更多实验级说明见 `exps/README.md` 与 `docs/exps-commons.md`。
