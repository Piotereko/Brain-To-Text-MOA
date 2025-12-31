import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
df_val = pd.read_csv('data/t15_pretrained_rnn_baseline/detailed_results_val_20251224_100649.csv')

# Calculate WER per trial
df_val['wer'] = (df_val['edit_distance'] / df_val['num_words']) * 100
df_val['per'] = (df_val['p_edit_distance'] / df_val['num_phonemes']) * 100

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. WER and PER by Corpus
ax1 = axes[0, 0]
corpus_stats = []
for corpus in df_val['corpus'].unique():
    df_corpus = df_val[df_val['corpus'] == corpus]
    wer = (df_corpus['edit_distance'].sum() / df_corpus['num_words'].sum()) * 100
    per = (df_corpus['p_edit_distance'].sum() / df_corpus['num_phonemes'].sum()) * 100
    corpus_stats.append({'Corpus': corpus, 'WER': wer, 'PER': per})

corpus_df = pd.DataFrame(corpus_stats)
x = np.arange(len(corpus_df))
width = 0.35
ax1.bar(x - width/2, corpus_df['WER'], width, label='WER', alpha=0.8, color='steelblue')
ax1.bar(x + width/2, corpus_df['PER'], width, label='PER', alpha=0.8, color='coral')
ax1.set_xlabel('Corpus', fontsize=11)
ax1.set_ylabel('Error Rate (%)', fontsize=11)
ax1.set_title('Error Rates by Corpus', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(corpus_df['Corpus'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. WER Distribution
ax2 = axes[0, 1]
ax2.hist(df_val['wer'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(df_val['wer'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_val["wer"].mean():.1f}%')
ax2.axvline(df_val['wer'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_val["wer"].median():.1f}%')
ax2.set_xlabel('Word Error Rate (%)', fontsize=11)
ax2.set_ylabel('Number of Trials', fontsize=11)
ax2.set_title('WER Distribution', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. PER Distribution
ax3 = axes[1, 0]
ax3.hist(df_val['per'], bins=50, color='coral', alpha=0.7, edgecolor='black')
ax3.axvline(df_val['per'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_val["per"].mean():.1f}%')
ax3.axvline(df_val['per'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_val["per"].median():.1f}%')
ax3.set_xlabel('Phoneme Error Rate (%)', fontsize=11)
ax3.set_ylabel('Number of Trials', fontsize=11)
ax3.set_title('PER Distribution', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Error categories
ax4 = axes[1, 1]
perfect = len(df_val[df_val['wer'] == 0])
low_error = len(df_val[(df_val['wer'] > 0) & (df_val['wer'] <= 10)])
medium_error = len(df_val[(df_val['wer'] > 10) & (df_val['wer'] <= 30)])
high_error = len(df_val[df_val['wer'] > 30])

categories = ['Perfect\n(0%)', 'Low\n(0-10%)', 'Medium\n(10-30%)', 'High\n(>30%)']
counts = [perfect, low_error, medium_error, high_error]
colors = ['green', 'yellowgreen', 'orange', 'red']

ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Number of Trials', fontsize=11)
ax4.set_title('WER Categories', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add percentage labels on bars
for i, (cat, count) in enumerate(zip(categories, counts)):
    percentage = (count / len(df_val)) * 100
    ax4.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('baseline_evaluation_summary.png', dpi=300, bbox_inches='tight')
print('\n✓ Visualization saved to: baseline_evaluation_summary.png')
plt.show()

# Print summary statistics
print('\n' + '='*70)
print('SUMMARY STATISTICS')
print('='*70)
print(f'\nTotal Validation Trials: {len(df_val)}')
print(f'\nOverall Metrics:')
print(f'  Average WER: {df_val["wer"].mean():.2f}%')
print(f'  Median WER:  {df_val["wer"].median():.2f}%')
print(f'  Average PER: {df_val["per"].mean():.2f}%')
print(f'  Median PER:  {df_val["per"].median():.2f}%')

print(f'\nWER Distribution:')
print(f'  Perfect (0%):      {perfect} trials ({perfect/len(df_val)*100:.1f}%)')
print(f'  Low (0-10%):       {low_error} trials ({low_error/len(df_val)*100:.1f}%)')
print(f'  Medium (10-30%):   {medium_error} trials ({medium_error/len(df_val)*100:.1f}%)')
print(f'  High (>30%):       {high_error} trials ({high_error/len(df_val)*100:.1f}%)')

print(f'\nCorpus Breakdown:')
for corpus in df_val['corpus'].unique():
    count = len(df_val[df_val['corpus'] == corpus])
    print(f'  {corpus}: {count} trials ({count/len(df_val)*100:.1f}%)')

print('\n' + '='*70)
