## 实验日志

所有的结果都是在bfclv4的mutil turn base上进行5次实验

目前看来，全量rl和全量sft都有效果且效果不错

### 发现的问题

- full5和full9应该是相同的实验，但是结果差很多
- full10是14k数据shuffle后sample 1k数据，结果竟然只比full4差一点
- 不管是full8还是full11，4bthinking在sft后会比基座低很多分
- lora总是完全无效
- 8b模型在评测的时候总是会报错
- 对于4binstruct model，hardgen 14k dataset，sft和rl差不多甚至比rl更好

### 260210

lora1，lora2，verl1，verl2，full1，full2的实验都有问题，因为构造的openai格式的messages数据集，工具调用相关的系统提示词没有和qwen3一模一样

复现美团的gem文章，合成了5k数据，进行以下实验

- full3，4b-instruct，gem 5k（token length的mean是1.9k，std是0.7k），sft，12.7%
- verl2，4b-instruct，gem 5k，grpo，reward_fn，0.4%
- verl5，4b-instruct，gem 2k（token length的mean是2.5k,std是0.4k），grpo，reward_fn，2%

凡是使用gem合成的数据，效果都很差

正确的系统提示词模版，全量hardgen数据，进行以下实验

- verl4，4b-instruct，grpo，reward_fn，42.6%
- full4，4b-instruct，sft，47.3%

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
- verl11，4b-thinking，egpo，reward_fn_egpo，62.1%（高于grpo）
- verl12，4b-instruct，grpo，reward_fn_grpo，29.1%（提升很少）
- full10，4b-instruct，sft，45.6%（结果很好）
- full11，4b-thinking，sft，36.2%

为了再次确认基座模型的得分，再跑一遍基座的评测

- baseline_4b，43%
- baseline_4binstruct，26.1%
- baseline_4bthinking，58.1%

之后再跑跑hardgen_1k和hardgen_1k_shuffle的实验

确定egpo有效（不一定要比grpo高）之后，就可以跑全量的hardgen了

### 260303

对比不同奖励函数的影响

- verl13，4b-instruct，hardgen_1k_shuffle，grpo，reward_fn，

在hardgen_1k上重跑grpo和egpo

- verl14，4b-thinking，grpo，reward_fn_grpo，
- verl15，4b-thinking，egpo，reward_fn_egpo，

hardgen全量数据实验

- verl16，4b-thinking，grpo，reward_fn_grpo，
- verl17，4b-thinking，egpo，reward_fn_egpo，
- verl18，4b-thinking，grpo，reward_fn，
