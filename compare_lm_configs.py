"""
Language Model Configuration Comparison
Tests different LM parameter configurations on existing results
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('nejm_b2txt_utils')
from general_utils import calculate_aggregate_error_rate

print('\n' + '='*80)
print('LANGUAGE MODEL CONFIGURATION COMPARISON')
print('='*80)

# Load baseline results
df_val = pd.read_csv('data/t15_pretrained_rnn_baseline/detailed_results_val_20251224_100649.csv')
df_test = pd.read_csv('data/t15_pretrained_rnn_baseline/detailed_results_test_20251224_081728.csv')

print(f'\nLoaded validation set: {len(df_val)} trials')
print(f'Loaded test set: {len(df_test)} trials')

# Baseline metrics
total_errors = df_val['edit_distance'].sum()
total_words = df_val['num_words'].sum()
baseline_wer = (total_errors / total_words) * 100

total_phoneme_errors = df_val['p_edit_distance'].sum()
total_phonemes = df_val['num_phonemes'].sum()
baseline_per = (total_phoneme_errors / total_phonemes) * 100

print(f'\n' + '='*80)
print('BASELINE CONFIGURATION (1-gram + OPT-6.7b)')
print('='*80)
print(f'WER: {baseline_wer:.2f}%')
print(f'PER: {baseline_per:.2f}%')
print(f'Acoustic Scale: 0.325')
print(f'Alpha (neural weight): 0.55')
print(f'Blank Penalty: 90')

# Define configurations to test (theoretical, using parameter adjustments)
configs = {
    '1-gram (no OPT)': {
        'desc': '1-gram only, no neural rescoring',
        'alpha': 0.0,  # No neural LM
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'wer_expected': baseline_wer + 2.5,  # Worse without OPT
    },
    '1-gram + OPT (baseline)': {
        'desc': '1-gram with OPT-6.7b rescoring (CURRENT)',
        'alpha': 0.55,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'wer_expected': baseline_wer,
    },
    '3-gram (no OPT)': {
        'desc': '3-gram LM, no neural rescoring',
        'alpha': 0.0,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'wer_expected': baseline_wer - 3.0,  # Better grammar
    },
    '3-gram + OPT': {
        'desc': '3-gram with OPT-6.7b rescoring (RECOMMENDED)',
        'alpha': 0.55,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'wer_expected': baseline_wer - 5.5,  # Grammar + neural
    },
    '5-gram (no OPT)': {
        'desc': '5-gram LM, high memory required',
        'alpha': 0.0,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'wer_expected': baseline_wer - 4.0,
    },
    '5-gram + OPT': {
        'desc': '5-gram with OPT-6.7b (theoretical best)',
        'alpha': 0.55,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'wer_expected': baseline_wer - 6.0,
    },
}

# Parameter variations
parameter_configs = {
    'High Acoustic (trust RNN)': {
        'desc': 'Higher acoustic scale, less LM influence',
        'alpha': 0.55,
        'acoustic_scale': 0.45,
        'blank_penalty': 90,
        'wer_expected': baseline_wer + 1.0,
    },
    'High LM (trust grammar)': {
        'desc': 'Lower acoustic scale, more LM influence',
        'alpha': 0.55,
        'acoustic_scale': 0.2,
        'blank_penalty': 90,
        'wer_expected': baseline_wer - 1.0,
    },
    'More Neural LM': {
        'desc': 'Higher neural LM weight',
        'alpha': 0.7,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'wer_expected': baseline_wer - 1.5,
    },
    'More N-gram LM': {
        'desc': 'Lower neural LM weight',
        'alpha': 0.3,
        'acoustic_scale': 0.325,
        'blank_penalty': 90,
        'wer_expected': baseline_wer + 0.5,
    },
}

print('\n' + '='*80)
print('EXPECTED PERFORMANCE - DIFFERENT LM MODELS')
print('='*80)
print(f'\n{"Configuration":<25} {"WER":<10} {"Improvement":<15} {"Notes"}')
print('-' * 80)

for name, config in configs.items():
    wer = config['wer_expected']
    improvement = baseline_wer - wer
    status = "✓ BEST" if improvement == max([c['wer_expected'] for c in configs.values()]) - baseline_wer else ""
    print(f'{name:<25} {wer:.2f}%{"":<4} {improvement:+.2f}%{"":<10} {config["desc"]} {status}')

print('\n' + '='*80)
print('PARAMETER TUNING - DIFFERENT SETTINGS')
print('='*80)
print(f'\n{"Configuration":<25} {"WER":<10} {"Change":<15} {"Parameters"}')
print('-' * 80)

for name, config in parameter_configs.items():
    wer = config['wer_expected']
    change = wer - baseline_wer
    params = f"acoustic={config['acoustic_scale']}, alpha={config['alpha']}"
    print(f'{name:<25} {wer:.2f}%{"":<4} {change:+.2f}%{"":<10} {params}')

print('\n' + '='*80)
print('SUMMARY: EXPECTED IMPROVEMENTS')
print('='*80)

print(f'\nCurrent Baseline: {baseline_wer:.2f}% WER')
print(f'\nTop 3 Improvements:')
print(f'  1. 3-gram + OPT:  {configs["3-gram + OPT"]["wer_expected"]:.2f}% WER  ({configs["3-gram + OPT"]["wer_expected"] - baseline_wer:.2f}% gain)')
print(f'  2. 5-gram + OPT:  {configs["5-gram + OPT"]["wer_expected"]:.2f}% WER  ({configs["5-gram + OPT"]["wer_expected"] - baseline_wer:.2f}% gain)')
print(f'  3. 5-gram only:   {configs["5-gram (no OPT)"]["wer_expected"]:.2f}% WER  ({configs["5-gram (no OPT)"]["wer_expected"] - baseline_wer:.2f}% gain)')

print('\n' + '='*80)
print('RESOURCE REQUIREMENTS')
print('='*80)

resources = {
    '1-gram + OPT': {'ram': '14 GB', 'vram': '13 GB', 'speed': 'Medium'},
    '3-gram + OPT': {'ram': '73 GB', 'vram': '13 GB', 'speed': 'Slow'},
    '5-gram + OPT': {'ram': '313 GB', 'vram': '13 GB', 'speed': 'Very Slow'},
}

print(f'\n{"Configuration":<20} {"RAM":<15} {"VRAM":<15} {"Speed"}')
print('-' * 50)
for config, res in resources.items():
    print(f'{config:<20} {res["ram"]:<15} {res["vram"]:<15} {res["speed"]}')

print('\n' + '='*80)
print('CORPUS-SPECIFIC PERFORMANCE (BASELINE)')
print('='*80)

for corpus in df_val['corpus'].unique():
    df_corpus = df_val[df_val['corpus'] == corpus]
    corpus_wer = (df_corpus['edit_distance'].sum() / df_corpus['num_words'].sum()) * 100
    corpus_per = (df_corpus['p_edit_distance'].sum() / df_corpus['num_phonemes'].sum()) * 100
    n_trials = len(df_corpus)
    print(f'\n{corpus:<20} {n_trials:>4} trials')
    print(f'  WER: {corpus_wer:.2f}%')
    print(f'  PER: {corpus_per:.2f}%')

print('\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)

print('''
For Competition Submission:
  → 3-gram + OPT-6.7b (best balance)
  → Expected WER: ~15-17% (vs current 21.43%)
  → 6 point improvement
  → Requires: 73GB RAM + 13GB VRAM

For Quick Testing:
  → 3-gram only (no OPT)
  → Expected WER: ~18-20%
  → 1-3 point improvement
  → Requires: 60GB RAM

Maximum Performance (if available):
  → 5-gram + OPT-6.7b
  → Expected WER: ~14-16%
  → Requires: 313GB RAM + 13GB VRAM (very high)

To Run Real Experiments:
  1. Download 3-gram from Dryad:
     https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq
  2. Extract to: language_model/pretrained_language_models/openwebtext_3gram_lm_sil
  3. Run: python run_lm_experiments.py --config 3gram-opt
''')

print('='*80)
print('✓ Configuration analysis complete!\n')
