## 实验日志

所有的结果都是在bfclv4的mutil turn base上进行5次实验

目前看来，全量rl有效果且效果不错

### 26-02-10

lora1，lora2，verl1，verl2，full1，full2的实验似乎都有问题，因为构造的openai格式的messages数据集，工具调用相关的系统提示词没有和qwen3一模一样

新增verl3和full3，使用复现美团的文章合成的五千多条数据进行实验

新增verl4和full4，使用修复好的代码，重新进行实验

发现full4实验有效，新增lora3实验

对于hargen 14k数据，token length的mean是6.1k，std是1.4k

对于gem 5k，token length的mean是1.9k，std是0.7k

或许这是导致使用gem 5k数据的强化学习和监督微调效果都很差的原因？

新增verl5实验，使用gem 5k过滤后的gem 2k数据，gem 2k的token length的mean是2.5k,std是0.4k

凡是使用gem合成的数据，效果都很差，或许问题就是出在数据质量上？

### 26-02-12

lora3无效，或许说明lora就是无效？

verl4的结果，比full4的结果差就算了，为什么会比verl1的效果差呢？

### 26-02-27

目前看来，如果像构造高质量的数据，离不开昂贵的api，究其本质，更接近蒸馏

所以还是觉得RL更有搞头（也许）

0.6B的模型在bfclv4 multi turn base上得分为0，看来能力太差的模型就是不行啊

现在计划直接上dashscope的api，合成1k数据，在此之前，先测试1k数据是否有效

full5，在hardgen 1k上full sft，有效

full6，在hardgen 100上full sft，有效

full7，在looptool 1k上full sft，几乎无效，甚至可能下降

verl6，在looptool 1k上full rl，几乎无效，但没有下降

### 26-02-28

新增 **verl7**（GRPO）、**verl8**（EGPO），数据均为 hardgen 1k（convert 产出 hardgen_1k/train.parquet、test.parquet）。两实验同数据、同奖励、同参，仅 adv 不同；奖励用 reward_fn_egpo 严格二元。跑法：先执行一次 convert，再分别 run_train_only.sh，见 exps/verl7、exps/verl8 的 README。

verl7/verl8 基座为 Qwen3-4B-**Thinking**-2507（EGPO 用 CoT 熵需思考模型）。reward_fn_egpo 先校验 think 块格式，再对 think 块后内容做 tool_call AST 校验，全对才 1 分。另新增 **baseline_4bthinking**（4B-Thinking 零样本）、**baseline_8b**（8B 零样本），见各自 README。

新增 **verl9**（GRPO + hardgen 1k），与 verl7/8 同数据、同参，基座 Qwen3-4B-**Instruct**-2507，奖励 reward_fn_grpo（不要求 think 块）。见 exps/verl9/README.md。

新增 **full8**、**full9**（全量 SFT，hardgen 1k），与 RL 同数据：train.parquet→train.jsonl、test.parquet→test.jsonl 由 parquet_to_openai_messages 导出，SFT 中 dataset=train.jsonl、val_dataset=test.jsonl。full8 基座 Qwen3-4B-Thinking，full9 基座 Qwen3-4B-Instruct。见 exps/full8、exps/full9 的 README。

**verl7、verl8、full9 已完成**，实验结果与简要分析见下方「实验结果」及本节「26-02-28 分析」。verl9、full8 未完成，待跑。

#### 分析

- **baseline 参考**：baseline_4binstruct（用于对照，4B-Instruct 零样本）multi turn base 约 26.5%～27.5%，均值约 27%；baseline_4bthinking（4B-Thinking 零样本）multi turn base 约 59%～63%。
- **full9 与 baseline 对比**：full9（4B-Instruct + hardgen 1k 全量 SFT）5 轮 multi turn base 均值约 25.8%，略低于 baseline_4binstruct（约 27%），即当前与 RL 同数据、同导出的 SFT（Instruct）未超过零样本 baseline_4binstruct。
- **full5 与 full9 对比（均为 hardgen 1k + 4B-Instruct 全量 SFT，训练参数几乎一致）**：full5 多轮 base 约 38.00%、38.50%、38.00%、37.00%、37.50%，均值约 **37.8%**，高于 baseline_4binstruct；full9 均值约 **25.8%**，低于 baseline_4binstruct。主要差异在**数据来源**：full5 使用 hardgen 原始 json 取 1000 条（`hardgen_openai_messages_fc.json#1000`）+ `split_dataset_ratio: 0.05` 做 train/val 划分；full9 使用与 verl7/verl8 同源的 convert 产出 parquet，再导出为 train.jsonl / test.jsonl。即同一「hardgen 1k」因数据划分与预处理流程（直接 json vs convert→parquet→jsonl）不同，可能导致训练集/评估分布不一致，从而结果差异大，值得后续对齐数据与复现。
- **full6**：为 hardgen **100** 条数据 SFT（非 1k），5 轮 multi turn base 均值约 30.5%，高于 baseline_4binstruct；文档中若曾误标为 1k 已修正。
- **GRPO vs EGPO（同数据、同奖励、同基座 Thinking）**：verl7（GRPO）5 轮 multi turn base 均值约 61.7%；verl8（EGPO）约 61.9%。两者水平接近，EGPO 略稳，未体现明显优势。
- **RL vs SFT（同数据 hardgen 1k）**：verl7/verl8（RL + Thinking）约 62%。需待 full8（SFT + Thinking）完成后对比 SFT 在 Thinking 基座上的表现。

### 26-03-01

**重要修正**：发现之前实现的 EGPO 算法有误。错误实现使用了**整个 response 的熵**来计算 EGPO advantage，而根据论文规范，EGPO 应该只使用 **Chain-of-Thought (CoT) 部分的熵**。

**错误影响范围**：
- **verl8**（EGPO 实验）：使用了错误的算法实现，实验结果（multi turn base 均值约 61.9%）基于错误的熵计算，需要重新实验验证正确实现的效果。

**修正内容**：
- 已修复 EGPO 实现，现在使用 Qwen3 的 `<think>` (token ID 151667) 和 `</think>` (token ID 151668) token 标记来识别 CoT 区域
- 只计算 CoT 部分的平均 token 熵，而不是整个 response 的熵
- 添加了 `_create_cot_mask_from_redacted_reasoning()` 辅助函数来创建 CoT mask
- 移除了向后兼容逻辑，确保算法正确性（如果缺少 `responses` 字段会报错终止）

**后续计划**：
- 需要重新运行 verl8 实验，使用修正后的 EGPO 算法，验证正确实现的效果

### 26-03-02

在hardgen 14k随机采样出1k数据得到hardgen_1k_shuffle

在hardgen_1k_shuffle上进行以下实验

verl10，4b-thinking，grpo，reward_fn_grpo
verl11，4b-thinking，egpo，reward_fn_egpo
verl12，4b-instruct，grpo，reward_fn_grpo
full11，4b-thinking，sft
full10，4b-instruct，sft

实验结果见 exps/RESULT.csv。

**full10、full11、verl10 已完成**（26-03-02 数据 hardgen_1k_shuffle，5 轮 multi turn base；对比均为**同基座**零样本 baseline）：
- **verl10**（4B-Thinking + GRPO + reward_fn_grpo）：均值 **60.5%**（62.5, 59.5, 62, 59.5, 60），与同基座 **baseline_4bthinking**（约 60%）持平略高。
- **full10**（4B-Instruct + SFT）：均值 **45.6%**（45.5, 46.5, 44.5, 46.5, 45），明显高于同基座 **baseline_4binstruct**（约 27%）。
- **full11**（4B-Thinking + SFT）：均值 **36.2%**（36.5, 35.5, 36, 37, 36），明显低于同基座 **baseline_4bthinking**（约 60%），即 SFT 在该数据上拉低了 Thinking 零样本表现；同数据下 verl10（GRPO）60.5% 保持基座水平。

**full8 重跑评测说明**：full8 训练基座为 4B-Thinking，此前 merge 脚本误用 4B-Instruct 作为 BASE_MODEL_PATH，导致合并目录中 tokenizer/config 来自 Instruct，eval_config5 的 remote_openai_tokenizer_path 也为 Instruct。二者均可能导致评测异常差（当前约 6.9%）。已修复 eval_config5；合并目录需用 Thinking 基座重新覆盖 tokenizer 与 config 后**重跑评测**。操作示例：`MERGED_DIR=/dfs/data/models/hardtry-4b-full8`，从 `Qwen3-4B-Thinking-2507` 拷贝 tokenizer.json、tokenizer_config.json、config.json、configuration.json、generation_config.json 等至 `$MERGED_DIR`，再重新执行 eval 流程。

**基座/合并/评测一致性检查（全量）**：

- **合并（merge）**  
  - **Verl**：合并不使用 base 模型路径，无「指定错基座」问题。  
  - **Swift**：已逐实验核对。full10、full11 与模板已改为从 `sft_config.yaml` 的 `model` 读取 BASE_MODEL_PATH；full8、full9 与 full1–7、lora3 的 merge 脚本中 BASE_MODEL_PATH 与各自训练基座（sft_config 或 conf/sft_config）一致。**唯一已知历史问题**：full8 若在修复前已跑过合并，则合并目录中 tokenizer/config 可能仍来自 Instruct，需按上文覆盖 Thinking 后重跑评测；当前 full8 的 merge 脚本已为 Thinking，之后再合并则正确。

- **评测（eval）**  
  - 已核对所有含 eval_config5 的实验，`remote_openai_tokenizer_path` 与实验基座一致：  
    - 4B：baseline_4b → Qwen3-4B  
    - 4B-Instruct：baseline_4binstruct、full1–7/9/10、lora1–3、verl1–6/9/12 → Qwen3-4B-Instruct-2507  
    - 4B-Thinking：baseline_4bthinking、full8/11、verl7/8/10/11 → Qwen3-4B-Thinking-2507  
    - 8B：baseline_8b → Qwen3-8B  
  - 未发现其它实验存在「评测指定错基座」。
