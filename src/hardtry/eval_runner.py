import os
import sys
import shutil
import time
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvalArguments:
    # --- 核心任务配置 ---
    model_name: str = field(metadata={"help": "Model name for bfcl (e.g., Qwen/Qwen3-4B-FC)"})
    test_category: str = field(default="multi_turn_base", metadata={"help": "Test category for bfcl"})
    
    # --- 运行环境配置 ---
    venv_activate_path: str = field(default="/dfs/data/uv-venv/gorilla/bin/activate", metadata={"help": "Path to venv activate script"})
    remote_openai_tokenizer_path: str = field(default="", metadata={"help": "Path to local tokenizer for vllm backend"})
    remote_openai_base_url: str = field(default="http://localhost:8000/v1")
    remote_openai_api_key: str = field(default="EMPTY")
    
    # --- 并行控制 ---
    num_runs: int = field(default=5, metadata={"help": "Total number of parallel runs"})
    threads_per_run: int = field(default=32, metadata={"help": "Number of threads for each bfcl process"})
    
    # --- 路径控制 ---
    base_artifact_dir: str = field(
        default="./eval_results", 
        metadata={"help": "Root directory to store all run logs and results"}
    )
    experiment_name: str = field(
        default="default_exp", 
        metadata={"help": "Name of the experiment to create subfolder"}
    )
    # 指定收集结果的目标文件夹
    summary_output_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "Directory to copy the collected CSV results. If None, defaults to <output_dir>/summary_csvs"}
    )

class ParallelEvalRunner:
    def __init__(self, args: EvalArguments):
        self.args = args
        
        # 构造实验的根输出目录: base_artifact_dir/experiment_name
        self.output_dir = os.path.join(self.args.base_artifact_dir, self.args.experiment_name)
        
        # 构造环境变量
        self.env_vars = {
            "REMOTE_OPENAI_BASE_URL": self.args.remote_openai_base_url,
            "REMOTE_OPENAI_API_KEY": self.args.remote_openai_api_key,
            "REMOTE_OPENAI_TOKENIZER_PATH": self.args.remote_openai_tokenizer_path,
            "PATH": os.environ.get("PATH", "")
        }

    def run_single_eval(self, run_id):
        """单个实验任务"""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        
        # 具体的某一次运行目录
        run_dir = os.path.join(self.output_dir, f"run_{run_id}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        log_file_path = os.path.join(run_dir, "experiment.log")
        
        # 构造命令
        cmd = f"""
        source {self.args.venv_activate_path}
        echo "--- Start Generation [Run {run_id}] ---"
        bfcl generate --model "{self.args.model_name}" --test-category "{self.args.test_category}" --backend vllm --skip-server-setup --num-threads "{self.args.threads_per_run}" --result-dir "{run_dir}/result"
        echo "--- Start Evaluation [Run {run_id}] ---"
        bfcl evaluate --model "{self.args.model_name}" --test-category "{self.args.test_category}" --result-dir "{run_dir}/result" --score-dir "{run_dir}/score"
        """

        logger.info(f"🚀 [Run {run_id}] Started. Log: {log_file_path}")
        
        with open(log_file_path, "w") as log_file:
            try:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    env={**os.environ, **self.env_vars},
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"✅ [Run {run_id}] Completed successfully.")
                    return True, run_id, run_dir
                else:
                    logger.error(f"❌ [Run {run_id}] Failed. Check log.")
                    return False, run_id, run_dir
            except Exception as e:
                logger.error(f"💥 [Run {run_id}] Exception: {e}")
                return False, run_id, run_dir

    def collect_results(self):
        """收集并重命名 csv 结果文件"""
        
        # 判断用户是否指定了汇总路径
        if self.args.summary_output_dir:
            # 如果用户指定了路径，就用用户的
            dest_dir = self.args.summary_output_dir
        else:
            # 如果没指定，默认放在实验目录下的 summary_csvs
            dest_dir = os.path.join(self.output_dir, "summary_csvs")
            
        os.makedirs(dest_dir, exist_ok=True)
        
        logger.info(f"\n📦 [Collection] Collecting results to: {dest_dir}")

        if not os.path.exists(self.output_dir):
            logger.warning("❌ Artifact directory does not exist.")
            return

        count = 0
        for folder_name in sorted(os.listdir(self.output_dir)):
            run_path = os.path.join(self.output_dir, folder_name)
            
            # 确保是文件夹且以 run_ 开头
            if os.path.isdir(run_path) and folder_name.startswith("run_"):
                # 原始文件路径
                src_file = os.path.join(run_path, "score", "data_multi_turn.csv")
                
                if os.path.exists(src_file):
                    new_filename = f"data_multi_turn_{folder_name}.csv"
                    dest_file = os.path.join(dest_dir, new_filename)
                    
                    try:
                        shutil.copy(src_file, dest_file)
                        logger.info(f"  -> Copied: {new_filename}")
                        count += 1
                    except Exception as e:
                        logger.error(f"  ❌ Copy failed {folder_name}: {e}")
                else:
                    logger.warning(f"  ⚠️ File not found: {src_file}")

        logger.info(f"✅ Collection complete. Copied {count} files.\n")

    def run(self):
        print("=======================================================")
        print(f"🔥 Parallel Evaluation ({self.args.num_runs} runs)")
        print(f"🤖 Model: {self.args.model_name}")
        print(f"📂 Output Dir: {self.output_dir}")
        print("=======================================================\n")

        start_time = time.time()

        # 使用进程池
        with ProcessPoolExecutor(max_workers=self.args.num_runs) as executor:
            futures = executor.map(self.run_single_eval, range(1, self.args.num_runs + 1))
            results = list(futures)

        end_time = time.time()
        
        success_count = sum(1 for r in results if r[0])
        
        print(f"\n" + "="*55)
        print(f"🏁 All tasks finished!")
        print(f"Success: {success_count}/{self.args.num_runs}")
        print(f"Total Time: {(end_time - start_time)/60:.2f} mins")
        print("="*55)
        
        self.collect_results()

if __name__ == "__main__":
    parser = HfArgumentParser((EvalArguments,))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        eval_args, = parser.parse_yaml_file(yaml_file=sys.argv[1])
    else:
        eval_args, = parser.parse_args_into_dataclasses()

    runner = ParallelEvalRunner(eval_args)
    runner.run()