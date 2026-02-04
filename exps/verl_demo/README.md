## 用0.6B的模型，4张3090的环境进行调试

### 发现

reward_fn的compute_score接收的solution_str是不包含"<|im_start|>assistant"和"<|im_end|>"的纯字符串
