# Language Model Decoding Parameters Guide

## Overview
This document describes the key decoding parameters for the Brain-to-Text language model pipeline and their impact on rule compliance and performance.

## Architecture
```
Neural Data → RNN/Transformer → Phoneme Logits → N-gram Decoder → Word Hypotheses → Neural LM Rescoring → Final Prediction
```

## Key Decoding Parameters

### 1. Acoustic Scale (`--acoustic_scale`)
**Range**: 0.1 - 0.5  
**Default**: 0.325  
**Purpose**: Controls the weight of the acoustic model (RNN) output vs language model

- **Higher values** (0.4-0.5): Trust acoustic model more, less smoothing from LM
  - More accurate when RNN is confident
  - May produce less grammatical output
  
- **Lower values** (0.1-0.2): Trust language model more
  - More grammatical output
  - May ignore acoustic evidence
  
**Rule Compliance**: Values too low (<0.1) may produce linguistically plausible but acoustically incorrect outputs

### 2. Blank Penalty (`--blank_penalty`)
**Range**: 50 - 150  
**Default**: 90  
**Purpose**: Penalty for inserting silence/blank tokens between phonemes

- **Higher values** (100-150): Fewer silence insertions
  - Faster decoding
  - May merge words incorrectly
  
- **Lower values** (50-80): More silence tokens
  - Clearer word boundaries
  - May over-segment words
  
**Rule Compliance**: Proper silence insertion is critical for valid phoneme-to-word mapping

### 3. Alpha (`--alpha`)
**Range**: 0.0 - 1.0  
**Default**: 0.55  
**Purpose**: Weight for neural LM (OPT-6.7b) rescoring vs n-gram LM

- **alpha = 0.0**: Only n-gram LM (faster, less memory)
- **alpha = 0.5-0.6**: Balanced (recommended)
- **alpha = 1.0**: Only neural LM (slower, more contextual)

Formula: `total_score = acoustic_scale * acoustic_score + (1-alpha) * ngram_score + alpha * neural_score`

**Rule Compliance**: Neural LM provides better long-range context but requires validation against vocabulary

### 4. N-best (`--nbest`)
**Range**: 1 - 500  
**Default**: 100  
**Purpose**: Number of hypothesis candidates to generate before rescoring

- **Lower values** (10-50): Faster, less diversity
- **Higher values** (100-200): More candidates for rescoring, better but slower

**Rule Compliance**: More hypotheses increase chance of finding rule-compliant output

### 5. Beam Width (`--beam`)
**Range**: 5.0 - 25.0  
**Default**: 17.0  
**Purpose**: Beam search width for decoding (log-probability threshold)

- **Wider beam** (20-25): More thorough search, slower
- **Narrower beam** (10-15): Faster, may miss good hypotheses

### 6. Max Active States (`--max_active`)
**Range**: 1000 - 10000  
**Default**: 7000  
**Purpose**: Maximum active states in FST decoder

- **Higher values**: More memory, more thorough search
- **Lower values**: Faster, less memory, may prune good paths

## Rule Compliance Validation

### Phoneme Sequence Validity
The decoder ensures valid phoneme sequences through:
1. **TLG.fst**: Finite State Transducer encoding Token-Lexicon-Grammar
   - `T`: Token (phoneme) to word mapping
   - `L`: Lexicon (valid phoneme sequences)
   - `G`: Grammar (n-gram language model)

2. **words.txt**: Vocabulary constraint file
   - Only words in this file can be output
   - Ensures closed-vocabulary decoding

### Validation Checks
```python
# Example validation
def is_valid_output(prediction, vocab_file='words.txt'):
    with open(vocab_file, 'r') as f:
        vocab = set(line.split()[0] for line in f)
    
    words = prediction.split()
    return all(word in vocab for word in words)
```

## Recommended Configurations

### 1. Fast Baseline (1-gram)
```bash
python language-model-standalone.py \\
    --lm_path pretrained_language_models/openwebtext_1gram_lm_sil \\
    --nbest 50 \\
    --acoustic_scale 0.325 \\
    --blank_penalty 90 \\
    --alpha 0.0 \\
    --beam 15.0
```
- **Speed**: Fast (~0.5s/trial)
- **Memory**: Low (~1GB)
- **WER**: Baseline
- **Use case**: Quick experiments, no grammar

### 2. Balanced (3-gram + OPT)
```bash
python language-model-standalone.py \\
    --lm_path pretrained_language_models/openwebtext_3gram_lm_sil \\
    --do_opt \\
    --nbest 100 \\
    --acoustic_scale 0.325 \\
    --blank_penalty 90 \\
    --alpha 0.55 \\
    --beam 17.0
```
- **Speed**: Medium (~2-3s/trial)
- **Memory**: Medium (~73GB: 60GB n-gram + 13GB OPT)
- **WER**: Best balance
- **Use case**: Production, competition submission

### 3. Best Performance (5-gram + OPT)
```bash
python language-model-standalone.py \\
    --lm_path pretrained_language_models/openwebtext_5gram_lm_sil \\
    --rescore \\
    --do_opt \\
    --nbest 100 \\
    --acoustic_scale 0.325 \\
    --blank_penalty 90 \\
    --alpha 0.55 \\
    --beam 17.0
```
- **Speed**: Slow (~5-10s/trial)
- **Memory**: Very high (~313GB: 300GB n-gram + 13GB OPT)
- **WER**: Lowest (theoretical best)
- **Use case**: Final submission, high-end hardware only

### 4. Acoustic-Heavy (Trust RNN more)
```bash
python language-model-standalone.py \\
    --lm_path pretrained_language_models/openwebtext_3gram_lm_sil \\
    --nbest 100 \\
    --acoustic_scale 0.45 \\
    --blank_penalty 90 \\
    --alpha 0.3 \\
    --beam 17.0
```
- **Use case**: When RNN is very confident (e.g., after strong data augmentation)

### 5. LM-Heavy (Trust language model more)
```bash
python language-model-standalone.py \\
    --lm_path pretrained_language_models/openwebtext_3gram_lm_sil \\
    --do_opt \\
    --nbest 100 \\
    --acoustic_scale 0.2 \\
    --blank_penalty 90 \\
    --alpha 0.7 \\
    --beam 17.0
```
- **Use case**: When RNN is noisy, prioritize grammatical output

## Parameter Tuning Strategy

### Grid Search Template
```python
acoustic_scales = [0.2, 0.325, 0.45]
alphas = [0.3, 0.55, 0.7]
blank_penalties = [70, 90, 110]

for acoustic_scale in acoustic_scales:
    for alpha in alphas:
        for blank_penalty in blank_penalties:
            # Run evaluation
            # Track WER
            # Select best configuration
```

### Quick Tuning (3-5 experiments)
1. Start with default: `acoustic_scale=0.325, alpha=0.55, blank_penalty=90`
2. Try higher acoustic: `acoustic_scale=0.4`
3. Try higher LM: `alpha=0.7`
4. Adjust blank: `blank_penalty=110`
5. Select best based on val WER

### Full Tuning (10-20 experiments)
- 3 acoustic scales × 3 alphas × 2 blank penalties = 18 configurations
- Recommended for final submission

## Linguistic Improvements by Configuration

### N-gram Model Improvements
| Feature | 1-gram | 3-gram | 5-gram |
|---------|--------|--------|--------|
| Grammar | ✗ | ✓✓ | ✓✓✓ |
| Context | ✗ | 2 words | 4 words |
| OOV handling | Basic | Good | Best |
| Memory | 1GB | 60GB | 300GB |

### Neural LM Improvements
- **Long-range dependencies**: Captures context beyond n-gram window
- **Semantic coherence**: Better topic consistency
- **Rare constructions**: Better handling of uncommon phrases
- **Cost**: +13GB memory, 2-3x slower

## Common Issues and Solutions

### Issue 1: High WER on random word sequences
**Cause**: LM expects natural language structure  
**Solution**: Reduce `alpha` to trust acoustic model more on non-linguistic data

### Issue 2: Output contains OOV words
**Cause**: Vocabulary file mismatch  
**Solution**: Verify `words.txt` matches training corpus vocabulary

### Issue 3: Too many silence tokens
**Cause**: `blank_penalty` too low  
**Solution**: Increase `blank_penalty` to 100-110

### Issue 4: Words merge together
**Cause**: `blank_penalty` too high  
**Solution**: Decrease `blank_penalty` to 70-80

### Issue 5: Slow decoding
**Cause**: Large beam width or n-gram model  
**Solution**: Reduce `beam` to 13-15, use smaller n-gram model

## Validation Checklist

- [ ] All output words exist in vocabulary file
- [ ] Phoneme sequences map to valid words via lexicon
- [ ] No impossible phoneme transitions
- [ ] Sentence boundaries properly marked with `SIL`
- [ ] WER on validation set meets target
- [ ] Decoding speed acceptable for inference
- [ ] Memory usage within hardware limits

## References
- Language model code: `language_model/language-model-standalone.py`
- Decoder implementation: `language_model/runtime/server/x86/`
- FST compilation: `language_model/tools/fst/`
- Evaluation script: `model_training/evaluate_model.py`
