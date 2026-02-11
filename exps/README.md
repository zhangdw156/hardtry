## 记录所有实验信息

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