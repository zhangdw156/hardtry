### 集中实验入口（Hydra）

通过 Hydra 指定任务配置文件；多个配置用 Hydra 的 `-m` 多跑（多次运行，每次一个配置）。

```bash
# 单个任务配置
uv run python -m hardtry.run config_file=exps/commons/configs/run_example.yaml

# 多个任务配置：Hydra -m 会按顺序多次运行，每次一个 config_file
uv run python -m hardtry.run -m config_file=exps/a.yaml,exps/b.yaml,exps/c.yaml
```

每个配置文件只需包含 `tasks: [...]`（module = uv run python -m，command = bash 等）。示例见 `exps/commons/configs/run_example.yaml`。  
约定：实验相关配置统一放在各实验目录的 `configs/` 下。

---

### 把ipynb导出为py

```bash
jupyter nbconvert --to script *.ipynb
```