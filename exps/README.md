## 实验日志

所有的结果都是在bfclv4的mutil turn base上进行5次实验

目前看来，全量rl和全量sft都有效果且效果不错

### 260210

lora1，lora2，verl1，verl2，full1，full2的实验都有问题，因为构造的openai格式的messages数据集，工具调用相关的系统提示词没有和qwen3一模一样

新增verl3和full3，使用复现美团的文章合成的五千多条数据进行实验

新增verl4和full4，使用修复好的代码，重新进行实验

发现full4实验有效，新增lora3实验

对于hargen 14k数据，token length的mean是6.1k，std是1.4k

对于gem 5k，token length的mean是1.9k，std是0.7k

或许这是导致使用gem 5k数据的强化学习和监督微调效果都很差的原因？

新增verl5实验，使用gem 5k过滤后的gem 2k数据，gem 2k的token length的mean是2.5k,std是0.4k

凡是使用gem合成的数据，效果都很差，或许问题就是出在数据质量上？

### 260212

lora3无效，或许说明lora就是无效？

verl4的结果，比full4的结果差就算了，为什么会比verl1的效果差呢？

### 260227

测试使用少量数据是否可以提高得分

- full5，4b-instruct，hardgen 1k，sft，37.8%
- full6，4b-instruct，hardgen 100，sft，30.5%
- full7，4b-instruct，looptool 1k，sft，26.2%
- verl6，4b-instruct，looptool 1k，grpo，reward_fn（细粒度），27.5%

少量hardgen数据也可以提高得分

### 260228

复现EGPO算法

使用hardgen 14k前1k数据hardgen_1k

- verl7，4b-thinking，grpo，reward_fn_egpo（old version），61.7%
- verl8，4b-thinking，egpo（wrong with all response entroy），reward_fn_egpo（old version），61.9%
- verl9，4b-instruct，grpo，reward_fn_grpo，25.2%（远低于对应基座得分）
- full8，4b-thinking，sft，6.9%（远低于对应基座得分）
- full9，4b-instruct，sft，25.8%（按理来说应该和full5实验结果差不多才对）

### 260301

**重要修正**：发现之前实现的 EGPO 算法有误。错误实现使用了**整个 response 的熵**来计算 EGPO advantage，而根据论文规范，EGPO 应该只使用 **Chain-of-Thought (CoT) 部分的熵**。

### 260302

在hardgen 14k随机采样出1k数据得到hardgen_1k_shuffle

- verl10，4b-thinking，grpo，reward_fn_grpo，60.5%
- verl11，4b-thinking，egpo，reward_fn_egpo，
- verl12，4b-instruct，grpo，reward_fn_grpo，
- full10，4b-instruct，sft，45.6%（结果很好）
- full11，4b-thinking，sft，36.2%
