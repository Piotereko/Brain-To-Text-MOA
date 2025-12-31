import pandas as pd
import numpy as np

df = pd.read_csv('data/t15_pretrained_rnn_baseline/detailed_results_val_20251224_100649.csv')

print('\n' + '='*70)
print('EXAMPLE PREDICTIONS')
print('='*70)

# Show 10 random examples
import random
random.seed(42)
samples = df.sample(10)

for idx, row in samples.iterrows():
    wer = (row['edit_distance'] / row['num_words']) * 100 if row['num_words'] > 0 else 0
    print(f'\nTrial {row["trial"]} - {row["corpus"]}')
    print(f'  True:  {row["true_sentence"]}')
    print(f'  Pred:  {row["pred_sentence"]}')
    print(f'  WER:   {wer:.1f}%')

# Show best predictions
print('\n' + '='*70)
print('BEST PREDICTIONS (0% WER)')
print('='*70)
perfect = df[df['edit_distance'] == 0].head(5)
for idx, row in perfect.iterrows():
    print(f'\nTrial {row["trial"]} - {row["corpus"]}')
    print(f'  Sentence: {row["true_sentence"]}')

# Show worst predictions
print('\n' + '='*70)
print('WORST PREDICTIONS (Highest WER)')
print('='*70)
df['wer'] = df['edit_distance'] / df['num_words']
worst = df.nlargest(5, 'wer')
for idx, row in worst.iterrows():
    wer = row['wer'] * 100
    print(f'\nTrial {row["trial"]} - {row["corpus"]} (WER: {wer:.1f}%)')
    print(f'  True:  {row["true_sentence"]}')
    print(f'  Pred:  {row["pred_sentence"]}')

print('\n' + '='*70)
