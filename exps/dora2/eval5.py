import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import shutil

# ================= 配置区 =================
MODEL_NAME = "Qwen/Qwen3-4B-FC"
TEST_CATEGORY = "multi_turn_base"
THREADS = 32  # 每个 bfcl 内部的线程数
NUM_RUNS = 5   # 并行运行的总次数

# 路径配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_DIR_NAME = os.path.basename(SCRIPT_DIR)
VENV_ACTIVATE = "/dfs/data/uv-venv/gorilla/bin/activate"
BASE_ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "../../eval_results", f"{CURRENT_DIR_NAME}_parallel_5runs")

# 环境变量
ENV_VARS = {
    "REMOTE_OPENAI_BASE_URL": "http://localhost:8000/v1",
    "REMOTE_OPENAI_API_KEY": "EMPTY",
    "REMOTE_OPENAI_TOKENIZER_PATH": "/dfs/data/models/sloop-4b_dora2",
    "PATH": os.environ.get("PATH", "") # 保持原有的 PATH
}

def run_single_eval(run_id):
    """单个实验任务"""
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = os.path.join(BASE_ARTIFACT_DIR, f"run_{run_id}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    log_file_path = os.path.join(run_dir, "experiment.log")
    
    # 构造命令（通过 bash -c 执行，确保能 source 环境）
    # 注意：bfcl 命令需要确保在你的 PATH 中或者在 venv 激活后可用
    cmd = f"""
    source {VENV_ACTIVATE}
    echo "--- Start Generation ---"
    bfcl generate --model "{MODEL_NAME}" --test-category "{TEST_CATEGORY}" --backend vllm --skip-server-setup --num-threads "{THREADS}" --result-dir "{run_dir}/result"
    echo "--- Start Evaluation ---"
    bfcl evaluate --model "{MODEL_NAME}" --test-category "{TEST_CATEGORY}" --result-dir "{run_dir}/result" --score-dir "{run_dir}/score"
    """

    print(f"🚀 [Run {run_id}] 已启动。日志记录至: {log_file_path}")
    
    with open(log_file_path, "w") as log_file:
        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                env={**os.environ, **ENV_VARS},
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ [Run {run_id}] 成功完成。")
                return True
            else:
                print(f"❌ [Run {run_id}] 失败，请检查日志。")
                return False
        except Exception as e:
            print(f"💥 [Run {run_id}] 抛出异常: {e}")
            return False

def collect_results():
    """收集并重命名 csv 结果文件"""
    # 定义目标目录: 当前脚本目录下的 eval5_results
    dest_dir = os.path.join(SCRIPT_DIR, "eval5_results")
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"\n📦 [Collection] 开始收集 CSV 结果到: {dest_dir}")

    if not os.path.exists(BASE_ARTIFACT_DIR):
        print("❌ 结果根目录不存在，无法收集。")
        return

    count = 0
    # 遍历 BASE_ARTIFACT_DIR 下的所有文件夹 (例如 run_1_0127_xxxx)
    for folder_name in sorted(os.listdir(BASE_ARTIFACT_DIR)):
        run_path = os.path.join(BASE_ARTIFACT_DIR, folder_name)
        
        # 确保是文件夹且以 run_ 开头
        if os.path.isdir(run_path) and folder_name.startswith("run_"):
            # 原始文件路径: .../run_x_xx/score/data_multi_turn.csv
            # 注意：根据 bfcl evaluate 命令，结果通常在 score 目录下
            src_file = os.path.join(run_path, "score", "data_multi_turn.csv")
            
            if os.path.exists(src_file):
                # 构造新文件名: data_multi_turn_run_1_0127_xxxx.csv
                # folder_name 本身就是 "run_1_0127_xxxx"
                new_filename = f"data_multi_turn_{folder_name}.csv"
                dest_file = os.path.join(dest_dir, new_filename)
                
                try:
                    shutil.copy(src_file, dest_file)
                    print(f"  -> 已复制: {new_filename}")
                    count += 1
                except Exception as e:
                    print(f"  ❌ 复制失败 {folder_name}: {e}")
            else:
                print(f"  ⚠️ 未找到文件: {src_file}")

    print(f"✅ 收集完成，共复制 {count} 个文件。\n")

def main():
    print("=======================================================")
    print(f"🔥 开始并行实验 (总计 {NUM_RUNS} 次)")
    print(f"🤖 模型: {MODEL_NAME}")
    print(f"📂 根目录: {BASE_ARTIFACT_DIR}")
    print("=======================================================\n")

    start_time = time.time()

    # 使用进程池实现并行
    with ProcessPoolExecutor(max_workers=NUM_RUNS) as executor:
        results = list(executor.map(run_single_eval, range(1, NUM_RUNS + 1)))

    end_time = time.time()
    
    success_count = sum(1 for r in results if r)
    print(f"\n" + "="*55)
    print(f"🏁 并行任务结束！")
    print(f"成功: {success_count}/{NUM_RUNS}")
    print(f"总耗时: {(end_time - start_time)/60:.2f} 分钟")
    print("="*55)
    collect_results()

if __name__ == "__main__":
    main()