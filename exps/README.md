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

以上 verl7、verl8、verl9、full8、full9 均未完成，待跑。

## 实验结果

### lora1(似乎没有效果)

> ms-swift

> SFT, 使用所有回复计算loss

> hardgen 14k

实验结果

- 26.50%
- 27.00%
- 27.00%
- 26.50%
- 25.50%

### lora2(似乎没有效果)

> ms-swift

> SFT, 使用最后一轮回复计算loss

> hardgen 14k

实验结果

- 26.50%
- 24.50%
- 26.50%
- 27.00%
- 25.50%

### verl1(有效)

> verl

> GRPO强化学习

> hardgen 14k

实验结果

- 45.00%
- 45.50%
- 45.50%
- 45.50%
- 47.00%

### verl2(几乎没有提升)

> verl

> 带lora的GRPO强化学习

> hardgen 14k

实验结果

- 26.00%
- 27.50%
- 26.50%
- 26.50%
- 26.50%

### full1(无效，甚至降低)

> ms-swift

> 全量SFT，所有回复计算loss

> hardgen 14k

实验结果

- 24.50%
- 23.50%
- 23.00%
- 24.00%
- 22.50%

### full2(效果最差)

> ms-swift

> 全量SFT，最后一轮回复计算loss

> hardgen 14k

实验结果

- 10.50%
- 11.00%
- 10.50%
- 10.50%
- 11.00%

### baseline

> 用于对照

实验结果

- 26.5%
- 26.5%
- 26.5%
- 27.5%
- 27.5%

### baseline_4bthinking

> Qwen3-4B-Thinking-2507 在 BFCL multi-turn base 上的零样本表现（无训练），见 exps/baseline_4bthinking/README.md

实验结果

- 15.00%
- 15.50%
- 14.25%
- 15.50%
- 14.75%

### full4

> ms-swift

> all response -> loss

> hardgen 14k

实验结果

- 47.00%
- 47.50%
- 47.50%
- 47.00%
- 47.50%

### full3

> ms-swift

> all response -> loss

> gem 5k

实验结果

- 13.00%
- 12.50%
- 13.00%
- 13.00%
- 12.00%

### verl3

> verl

> gem 5k

实验结果

- 0.50%
- 0.00%
- 0.50%
- 0.50%
- 0.50%

### verl5

> verl

> gem 2k

实验结果

- 2.00%
- 2.00%
- 2.00%
- 2.00%
- 2.00%

### lora3

> ms-swift, lora

> hardgen 14k

> all response -> loss

实验结果

- 26.50%
- 26.50%
- 26.00%
- 27.00%
- 26.50%

### verl4

> verl

> hardgen 14k

实验结果

- 43.50%
- 42.50%
- 41.50%
- 43.00%
- 42.50%

### full5

> ms-swift

> hardgen 100

> all response -> loss

实验结果

- 38.00%
- 38.00%
- 38.00%
- 37.00%
- 37.50%

### full6

> ms-swift

> hardgen 1k

> all response -> loss

实验结果

- 30.00%
- 30.00%
- 31.00%
- 30.50%
- 30.50%

### full7

> ms-swift

> looptool 1k

> all response -> loss

实验结果

- 25.00%
- 27.00%
- 25.50%
- 27.50%
- 26.00%

### verl6

> verl

> looptool 1k

实验结果

- 28.50%
- 27.00%
- 28.00%
- 27.50%
- 26.50%

### verl7（未完成）

> verl

> GRPO 对照组，hardgen 1k，与 verl8 同数据、同奖励（reward_fn_egpo 严格二元）、同参；基座 Qwen3-4B-Thinking。数据：先 convert 一次，再 run_train_only.sh。见 exps/verl7/README.md。

实验结果

- （未完成，待跑）

### verl8（未完成）

> verl

> EGPO，hardgen 1k，与 verl7 同数据、同奖励、同参，仅 adv_estimator 为 egpo；基座 Qwen3-4B-Thinking。见 exps/verl8/README.md。

实验结果

- （未完成，待跑）

### verl9（未完成）

> verl

> GRPO，hardgen 1k，与 verl7/8 同数据、同参；基座 **Qwen3-4B-Instruct**，奖励 reward_fn_grpo（不要求 think 块）。见 exps/verl9/README.md。

实验结果

- （未完成，待跑）

### full8（未完成）

> ms-swift

> 全量 SFT，hardgen 1k；与 RL 同数据（dataset=train.jsonl，val_dataset=test.jsonl，由 parquet 导出）；基座 Qwen3-4B-Thinking，template qwen3。见 exps/full8/README.md。

实验结果

- （未完成，待跑）

### full9（未完成）

> ms-swift

> 全量 SFT，hardgen 1k；与 RL 同数据（dataset=train.jsonl，val_dataset=test.jsonl）；基座 Qwen3-4B-Instruct，template qwen3_nothinking。见 exps/full9/README.md。

实验结果

- （未完成，待跑）
