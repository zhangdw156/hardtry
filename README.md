### 实验一条龙

各实验由 **run_local.sh** 统筹，按顺序执行 `scripts/` 下各步骤脚本。

```bash
# 在实验目录下执行（或 bash exps/verl6/run_local.sh）
cd exps/verl6 && bash run_local.sh
```

约定：配置在 `configs/`，步骤脚本在 `scripts/`，一个 sh 对应一个步骤，各脚本可从任意目录执行。

两段数据转换示例见 **docs/exps-commons.md**（convert_hardgen_to_messages → convert_messages_to_verl）。

详细使用说明见 **docs/**：`docs/exps-commons.md`（实验公共层）、`docs/src-hardtry.md`（hardtry 包）。

---

### 把ipynb导出为py

```bash
jupyter nbconvert --to script *.ipynb
```