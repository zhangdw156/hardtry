"""
集中实验入口：通过 Hydra 指定任务配置文件；支持多配置多跑。
每个配置文件一个实验（tasks 串行）；多个配置用 Hydra 的 multirun（-m 或 --multirun）。
两个 -m 不冲突：前一个是 python 的 -m（跑模块），后一个是 Hydra 的 -m（multirun 多跑）。

用法:
  uv run python -m hardtry.run config_file=exps/commons/configs/run_example.yaml
  uv run python -m hardtry.run -m config_file=exps/a.yaml,exps/b.yaml
  uv run python -m hardtry.run --multirun config_file=exps/a.yaml,exps/b.yaml
"""

import subprocess
import tempfile
from pathlib import Path

import hydra
from omegaconf import OmegaConf

# 无包内 config 目录，用临时默认配置供 Hydra 注入 config_file
_run_cfg_dir = Path(tempfile.mkdtemp(prefix="hardtry_hydra_"))
(_run_cfg_dir / "run.yaml").write_text("config_file: null\n", encoding="utf-8")


def _run_tasks(tasks: list, cwd: Path) -> int:
    """串行执行 tasks，返回退出码。"""
    for i, task in enumerate(tasks):
        task = OmegaConf.to_container(task, resolve=True)
        kind = task.get("type")
        if kind == "module":
            module = task.get("module")
            if not module:
                print(f"任务 {i + 1}: module 类型缺少 module 字段")
                return 1
            config = task.get("config")
            cmd = ["uv", "run", "python", "-m", module]
            if config:
                cmd.append(str(config))
            print(f"任务 {i + 1}/{len(tasks)}: uv run python -m {module} ...")
        elif kind == "command":
            command = task.get("command")
            if not command:
                print(f"任务 {i + 1}: command 类型缺少 command 字段")
                return 1
            if isinstance(command, str):
                command = [command]
            cmd = command
            print(f"任务 {i + 1}/{len(tasks)}: {' '.join(str(c) for c in cmd)}")
        else:
            print(f"任务 {i + 1}: 未知类型 {kind}，已跳过")
            continue

        ret = subprocess.run(cmd, cwd=cwd)
        if ret.returncode != 0:
            print(f"任务 {i + 1} 失败，退出码 {ret.returncode}")
            return ret.returncode
    return 0


@hydra.main(
    config_path=str(_run_cfg_dir),
    config_name="run",
    version_base="1.3",
)
def main(cfg):
    """Hydra 入口：cfg.config_file 指向一个任务配置 yaml，加载其 tasks 并执行。"""
    config_file = cfg.get("config_file")
    if not config_file:
        print("请通过 config_file=path/to.yaml 指定任务配置文件。")
        print("多配置多跑: -m config_file=a.yaml,b.yaml,c.yaml")
        return 1

    path = Path(config_file)
    if not path.is_file():
        print(f"错误: 文件不存在 {path}")
        return 1

    data = OmegaConf.load(path)
    tasks = data.get("tasks") or []
    if not tasks:
        print("该配置中 tasks 为空。")
        return 0

    return _run_tasks(tasks, Path.cwd())


if __name__ == "__main__":
    main()
