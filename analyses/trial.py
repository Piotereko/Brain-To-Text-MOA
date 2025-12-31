import subprocess
import threading

class Colors:
    CMD1 = '\033[95m'  # Pink
    CMD2 = '\033[35m'  # Purple
    RESET = '\033[0m'

def run_command(command, prefix, color=""):
    """Run a shell command and print output in real-time with colored prefix"""
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        executable='/bin/bash'
    )
    
    # Print output in real-time with colored prefix
    for line in process.stdout:
        print(f"{color}{prefix}:{Colors.RESET} {line}", end='')
    
    process.wait()
    return process.returncode

# Define your commands
cmd1 = (
    "cd /Brain-To-Text-MOA && "
    "alias conda='~/miniconda3/bin/conda' && "
    "source ~/miniconda3/bin/activate && "
    "(conda activate b2txt25_lm || bash setup_lm.sh) && "
    "conda activate b2txt25_lm && "
    "python language_model/language-model-standalone.py "
    "--lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil "
    "--do_opt --nbest 100 "
    "--acoustic_scale 0.35 "
    "--blank_penalty 90 "
    "--alpha 0.55 "
    "--redis_ip localhost "
    "--gpu_number 0"
)
cmd2 = (
    "sleep 10 && "
    "cd /Brain-To-Text-MOA && "
    "alias conda='~/miniconda3/bin/conda' && "
    "source ~/miniconda3/bin/activate && "
    "(conda activate b2txt25 || bash setup.sh) && "
    "conda activate b2txt25 && "
    "python model_training/evaluate_model.py "
    "--gpu_number 0 "
    "--model_path /transformer_v1 "
    "--eval_type val"
)


# cmd1 = "alias conda='~/miniconda3/bin/conda' && source ~/miniconda3/bin/activate && bash setup_lm.sh && conda activate b2txt25_lm && python language_model/language-model-standalone.py --lm_path language_model/pretrained_language_models/openwebtext_1gram_lm_sil --do_opt --nbest 100 --acoustic_scale 0.325 --blank_penalty 90 --alpha 0.55 --redis_ip localhost --gpu_number 0"
# cmd2 = "echo 'Sleeping for 15 minutes...' && sleep 900 && alias conda='~/miniconda3/bin/conda' && source ~/miniconda3/bin/activate && bash setup.sh && conda activate b2txt25 && python model_training/evaluate_model.py --gpu_number 1 --model_path /kaggle/input/brain-to-text-25/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/ --data_dir /kaggle/input/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final --csv_path ./data/t15_copyTaskData_description.csv --eval_type val"

# Create threads with colored prefixes
thread1 = threading.Thread(target=run_command, args=(cmd1, "cmd1", Colors.CMD1))
thread2 = threading.Thread(target=run_command, args=(cmd2, "cmd2", Colors.CMD2))

# Start both threads
thread1.start()
thread2.start()

# Wait for both to complete
thread1.join()
thread2.join()

print("Both commands completed!")