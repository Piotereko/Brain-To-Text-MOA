"""
Language Model Experiment Runner
Person C: Language Model Experiments & Decoding

This script automates running the Brain-to-Text pipeline with different language model
configurations to compare performance.

Usage:
    python run_lm_experiments.py --config baseline
    python run_lm_experiments.py --config all
"""

import os
import sys
import time
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd

# Configuration presets
LM_CONFIGS = {
    'baseline-1gram': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_1gram_lm_sil',
        'do_opt': False,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'alpha': 0.0,
        'beam': 17.0,
        'description': 'Baseline 1-gram LM without neural rescoring'
    },
    '1gram-opt': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_1gram_lm_sil',
        'do_opt': True,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'alpha': 0.55,
        'beam': 17.0,
        'description': '1-gram LM with OPT-6.7b rescoring'
    },
    '3gram': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_3gram_lm_sil',
        'do_opt': False,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'alpha': 0.0,
        'beam': 17.0,
        'description': '3-gram LM without neural rescoring'
    },
    '3gram-opt': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_3gram_lm_sil',
        'do_opt': True,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'alpha': 0.55,
        'beam': 17.0,
        'description': '3-gram LM with OPT-6.7b rescoring (RECOMMENDED)'
    },
    '5gram': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_5gram_lm_sil',
        'do_opt': False,
        'rescore': True,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'alpha': 0.0,
        'beam': 17.0,
        'description': '5-gram LM with rescoring, no neural'
    },
    '5gram-opt': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_5gram_lm_sil',
        'do_opt': True,
        'rescore': True,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'alpha': 0.55,
        'beam': 17.0,
        'description': '5-gram LM with rescoring and OPT-6.7b (HIGH MEMORY)'
    },
}

# Parameter tuning experiments
TUNING_CONFIGS = {
    'tune-acoustic-high': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_3gram_lm_sil',
        'do_opt': True,
        'nbest': 100,
        'acoustic_scale': 0.45,  # Trust RNN more
        'blank_penalty': 90,
        'alpha': 0.55,
        'beam': 17.0,
        'description': 'Higher acoustic weight (trust RNN more)'
    },
    'tune-acoustic-low': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_3gram_lm_sil',
        'do_opt': True,
        'nbest': 100,
        'acoustic_scale': 0.2,  # Trust LM more
        'blank_penalty': 90,
        'alpha': 0.55,
        'beam': 17.0,
        'description': 'Lower acoustic weight (trust LM more)'
    },
    'tune-lm-high': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_3gram_lm_sil',
        'do_opt': True,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'alpha': 0.7,  # More neural LM
        'beam': 17.0,
        'description': 'Higher neural LM weight'
    },
    'tune-lm-low': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_3gram_lm_sil',
        'do_opt': True,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'alpha': 0.3,  # Less neural LM
        'beam': 17.0,
        'description': 'Lower neural LM weight'
    },
    'tune-blank-high': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_3gram_lm_sil',
        'do_opt': True,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 110,  # Fewer silences
        'alpha': 0.55,
        'beam': 17.0,
        'description': 'Higher blank penalty (fewer silences)'
    },
    'tune-blank-low': {
        'lm_path': 'language_model/pretrained_language_models/openwebtext_3gram_lm_sil',
        'do_opt': True,
        'nbest': 100,
        'acoustic_scale': 0.325,
        'blank_penalty': 70,  # More silences
        'alpha': 0.55,
        'beam': 17.0,
        'description': 'Lower blank penalty (more silences)'
    },
}


def check_lm_exists(lm_path):
    """Check if language model directory exists."""
    if not os.path.exists(lm_path):
        print(f"❌ Language model not found: {lm_path}")
        print(f"   Please download the model or check the path.")
        return False
    
    # Check for required files
    required_files = ['TLG.fst', 'words.txt']
    for fname in required_files:
        if not os.path.exists(os.path.join(lm_path, fname)):
            print(f"❌ Required file not found: {fname} in {lm_path}")
            return False
    
    return True


def build_lm_command(config, gpu_number=0, redis_ip='localhost'):
    """Build the language model server command."""
    cmd = [
        'python', 'language_model/language-model-standalone.py',
        '--lm_path', config['lm_path'],
        '--nbest', str(config['nbest']),
        '--acoustic_scale', str(config['acoustic_scale']),
        '--blank_penalty', str(config['blank_penalty']),
        '--alpha', str(config['alpha']),
        '--beam', str(config['beam']),
        '--redis_ip', redis_ip,
        '--gpu_number', str(gpu_number),
    ]
    
    if config.get('do_opt', False):
        cmd.append('--do_opt')
    
    if config.get('rescore', False):
        cmd.append('--rescore')
    
    return cmd


def build_eval_command(model_path, data_dir, eval_type='val', gpu_number=1):
    """Build the evaluation command."""
    cmd = [
        'python', 'model_training/evaluate_model.py',
        '--model_path', model_path,
        '--data_dir', data_dir,
        '--eval_type', eval_type,
        '--gpu_number', str(gpu_number),
    ]
    return cmd


def run_experiment(config_name, config, args):
    """Run a single experiment with the given configuration."""
    print(f"\n{'='*70}")
    print(f"Running experiment: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}\n")
    
    # Check if LM exists
    if not check_lm_exists(config['lm_path']):
        print(f"⏭️  Skipping {config_name} - LM not available\n")
        return None
    
    # Build commands
    lm_cmd = build_lm_command(config, gpu_number=args.lm_gpu, redis_ip=args.redis_ip)
    eval_cmd = build_eval_command(
        args.model_path, 
        args.data_dir, 
        eval_type=args.eval_type,
        gpu_number=args.eval_gpu
    )
    
    print("LM command:")
    print(" ".join(lm_cmd))
    print("\nEvaluation command:")
    print(" ".join(eval_cmd))
    print()
    
    if args.dry_run:
        print("🔍 Dry run - not executing\n")
        return None
    
    # Start LM server
    print("🚀 Starting language model server...")
    lm_process = subprocess.Popen(
        lm_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for LM to initialize (look for connection message)
    print("⏳ Waiting for LM to initialize...")
    time.sleep(30)  # Give it time to load
    
    # Check if process is still running
    if lm_process.poll() is not None:
        stdout, stderr = lm_process.communicate()
        print("❌ LM server failed to start!")
        print("STDERR:", stderr)
        return None
    
    print("✓ LM server started\n")
    
    # Run evaluation
    print("🧪 Running evaluation...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            eval_cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        eval_time = time.time() - start_time
        
        print(f"✓ Evaluation complete in {eval_time:.1f}s\n")
        
        if result.returncode != 0:
            print("❌ Evaluation failed!")
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out!")
        eval_time = None
    
    finally:
        # Stop LM server
        print("🛑 Stopping LM server...")
        lm_process.terminate()
        try:
            lm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            lm_process.kill()
        print("✓ LM server stopped\n")
    
    # Find the generated results file
    result_pattern = f"detailed_results_{args.eval_type}_*.csv"
    result_files = list(Path(args.model_path).glob(result_pattern))
    
    if result_files:
        latest_result = max(result_files, key=os.path.getctime)
        print(f"📊 Results saved to: {latest_result.name}\n")
        
        # Optionally rename file to include config name
        new_name = latest_result.parent / f"detailed_results_{config_name}_{args.eval_type}.csv"
        if args.rename_results:
            os.rename(latest_result, new_name)
            print(f"✏️  Renamed to: {new_name.name}\n")
            latest_result = new_name
        
        return {
            'config_name': config_name,
            'config': config,
            'eval_time': eval_time,
            'result_file': str(latest_result)
        }
    else:
        print("⚠️  No result file found\n")
        return None


def main():
    parser = argparse.ArgumentParser(description='Run language model experiments')
    parser.add_argument('--config', type=str, default='baseline-1gram',
                        help='Configuration name or "all" to run all configs')
    parser.add_argument('--model_path', type=str, 
                        default='./data/t15_pretrained_rnn_baseline',
                        help='Path to pretrained model')
    parser.add_argument('--data_dir', type=str,
                        default='./data/hdf5_data_final',
                        help='Path to data directory')
    parser.add_argument('--eval_type', type=str, default='val',
                        choices=['val', 'test'],
                        help='Evaluation split')
    parser.add_argument('--lm_gpu', type=int, default=0,
                        help='GPU for language model')
    parser.add_argument('--eval_gpu', type=int, default=1,
                        help='GPU for evaluation model')
    parser.add_argument('--redis_ip', type=str, default='localhost',
                        help='Redis server IP')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')
    parser.add_argument('--rename_results', action='store_true',
                        help='Rename result files with config name')
    parser.add_argument('--include_tuning', action='store_true',
                        help='Include parameter tuning experiments')
    
    args = parser.parse_args()
    
    # Determine which configs to run
    if args.config == 'all':
        configs_to_run = LM_CONFIGS
        if args.include_tuning:
            configs_to_run = {**LM_CONFIGS, **TUNING_CONFIGS}
    elif args.config == 'tuning':
        configs_to_run = TUNING_CONFIGS
    else:
        all_configs = {**LM_CONFIGS, **TUNING_CONFIGS}
        if args.config not in all_configs:
            print(f"❌ Unknown config: {args.config}")
            print(f"\nAvailable configs:")
            for name in all_configs.keys():
                print(f"  - {name}")
            sys.exit(1)
        configs_to_run = {args.config: all_configs[args.config]}
    
    print(f"\n🔬 Language Model Experiment Runner")
    print(f"{'='*70}")
    print(f"Configurations to run: {len(configs_to_run)}")
    for name in configs_to_run.keys():
        print(f"  - {name}")
    print()
    
    # Check Redis is running (if not dry run)
    if not args.dry_run:
        try:
            import redis
            r = redis.Redis(host=args.redis_ip, port=6379)
            r.ping()
            print("✓ Redis server is running\n")
        except:
            print("❌ Redis server is not running!")
            print("   Start it with: redis-server")
            print("   Or run in dry-run mode: --dry_run\n")
            sys.exit(1)
    
    # Run experiments
    results = []
    for config_name, config in configs_to_run.items():
        result = run_experiment(config_name, config, args)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📋 Experiment Summary")
    print(f"{'='*70}\n")
    print(f"Completed: {len(results)}/{len(configs_to_run)} experiments\n")
    
    if results:
        print("Results:")
        for r in results:
            print(f"\n{r['config_name']}:")
            print(f"  Time: {r['eval_time']:.1f}s")
            print(f"  File: {Path(r['result_file']).name}")
    
    # Save experiment log
    if results and not args.dry_run:
        log_file = f"experiments_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Experiment log saved to: {log_file}\n")
    
    print("✓ All experiments complete!\n")


if __name__ == '__main__':
    main()
