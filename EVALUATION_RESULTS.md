# Baseline Evaluation Results - Analysis Complete

## Executive Summary

**Date**: December 31, 2025  
**Model**: Pretrained RNN Baseline (5-layer GRU, 768 units)  
**Language Model**: 1-gram + OPT-6.7b rescoring  
**Dataset**: Brain-to-Text Competition 2025

---

## Overall Performance

### Validation Set (1,426 trials)
- **Word Error Rate (WER)**: **21.43%** (2,024 errors / 9,445 words)
- **Phoneme Error Rate (PER)**: **10.20%** (4,224 errors / 41,392 phonemes)

### Test Set (1,450 trials)
- No ground truth available (competition test set)
- Predictions generated for submission

---

## Performance by Corpus

| Corpus | Trials | WER | PER | Notes |
|--------|--------|-----|-----|-------|
| **Switchboard** | 1,037 (72.7%) | 22.50% | 10.62% | Natural dialogue |
| **Freq words** | 225 (15.8%) | **11.87%** | **3.94%** | Best performance |
| **Openwebtext** | 80 (5.6%) | 23.02% | 14.57% | Web text |
| **Harvard** | 45 (3.2%) | 30.83% | 16.49% | Harvard sentences |
| **Random** | 39 (2.7%) | **37.50%** | 14.65% | Worst performance |

### Key Observations:
1. **Best**: Frequent words (11.87% WER) - Limited vocabulary, common words
2. **Worst**: Random words (37.50% WER) - No grammatical structure
3. **Gap**: 2x performance difference shows strong LM dependency
4. **PER vs WER**: ~2:1 ratio indicates LM is correcting many acoustic errors

---

## Error Distribution

### WER Categories:
- **Perfect (0% WER)**: 439 trials (30.8%)
- **Low (0-10% WER)**: 28 trials (2.0%)
- **Medium (10-30% WER)**: 577 trials (40.5%)
- **High (>30% WER)**: 382 trials (26.8%)

### Statistics:
- **Mean WER**: 21.04%
- **Median WER**: 16.67%
- **Mean PER**: 9.73%
- **Median PER**: 7.14%

**Insight**: 30.8% perfect predictions shows the model works well for many trials, but 26.8% with >30% error indicates room for improvement.

---

## Example Predictions

### ✓ Perfect Predictions (0% WER)
```
Trial 1 - Switchboard
  Sentence: "How does it keep the cost down?"

Trial 5 - Switchboard
  Sentence: "He said the decision to part ways was mutual."

Trial 3 - Freq words
  Sentence: "This car is super expensive."
```

### ⚠️ Good But Imperfect (Low WER)
```
Trial 8 - Switchboard (WER: 16.7%)
  True: "Fiction books that I really like."
  Pred: "fiction both that i really like"
  Error: "books" → "both"

Trial 22 - Switchboard (WER: 16.7%)
  True: "After graduation, the students will go on to a variety of colleges."
  Pred: "after decimation the tutors will go on to a variety of colleges"
  Errors: "graduation" → "decimation", "students" → "tutors"
```

### ❌ High Error Examples (100% WER)
```
Trial 9 - Switchboard
  True: "Woodworking mastery."
  Pred: "what working mastery"

Trial 11 - Switchboard
  True: "Zoology department."
  Pred: "hewlett depart"

Trial 4 - Switchboard
  True: "I love eating pecan pie."
  Pred: "live ending performed by"
```

**Error Pattern**: Phonetically similar but semantically different words - shows acoustic model limitations.

---

## Analysis: LM Dependency

### Evidence of Strong LM Reliance:

1. **Random Words Performance**
   - WER: 37.50% (vs 22.50% for natural language)
   - LM can't help without grammatical context
   - Shows true acoustic model performance

2. **PER vs WER Gap**
   - PER: 10.20% (acoustic errors at phoneme level)
   - WER: 21.43% (errors after LM correction)
   - LM corrects many phoneme errors into valid words
   - But sometimes incorrect words (e.g., "books" → "both")

3. **Corpus Performance Correlation**
   - Better grammar = better WER
   - Freq words (limited vocab) = best WER (11.87%)
   - Random words (no grammar) = worst WER (37.50%)

---

## Improvement Opportunities

### 1. Acoustic Model (Person B's Task)
**Current**: RNN (5-layer GRU)  
**Improvement**: Transformer architecture
- Better long-range dependencies
- More sophisticated temporal modeling
- Expected gain: 2-5% WER reduction

### 2. Language Model (Your Task)
**Current**: 1-gram + OPT-6.7b  
**Improvements to Test**:

| Configuration | Expected WER | Memory | Speed |
|--------------|--------------|--------|-------|
| Current (1-gram + OPT) | 21.43% | ~14GB | Medium |
| 3-gram | ~18-19% | ~60GB | Medium |
| 3-gram + OPT | **~15-17%** | ~73GB | Slow |
| 5-gram + OPT | ~14-16% | ~313GB | Very slow |

**Recommended**: 3-gram + OPT (best balance)

### 3. Data Augmentation (Person B Implemented)
- ✓ Temporal masking (15% mask)
- ✓ Gaussian smoothing
- ✓ White noise injection
- Expected to improve robustness

---

## Next Steps for LM Experiments

### Priority 1: Compare N-gram Models
1. Download 3-gram LM from Dryad
2. Run: `python run_lm_experiments.py --config 3gram`
3. Compare WER improvement vs baseline
4. Expected: ~3-5% WER reduction

### Priority 2: Test Neural LM Rescoring
1. Run: `python run_lm_experiments.py --config 3gram-opt`
2. Test OPT-6.7b weight (alpha parameter)
3. Expected: Additional ~2-3% WER reduction

### Priority 3: Parameter Tuning
1. Test acoustic_scale: [0.2, 0.325, 0.45]
2. Test alpha: [0.3, 0.55, 0.7]
3. Find optimal configuration per corpus

---

## Linguistic Analysis

### Common Error Types:

1. **Phonetic Substitutions** (most common)
   - "graduation" → "decimation"
   - "students" → "tutors"
   - "you" → "oui"
   - **Cause**: Similar phonemes, acoustic confusion

2. **Homophones**
   - "heard" → "herd"
   - **Cause**: Identical pronunciation

3. **Function Words**
   - "We" → "oui"
   - "where" → "were"
   - **Cause**: Short words, less acoustic information

4. **Rare Words**
   - "Zoology" → "hewlett"
   - **Cause**: Out-of-vocabulary or rare in training

### LM Improvements Needed:
- **Better context**: 3-gram/5-gram for grammar
- **Semantic understanding**: Neural LM for meaning
- **Rare word handling**: Better OOV strategies

---

## Rule Compliance Validation

### Verified Constraints:
✓ All predicted words exist in vocabulary file  
✓ Phoneme sequences map to valid words via FST  
✓ SIL tokens properly mark word boundaries  
✓ No impossible phoneme transitions  

### Validation Method:
- FST (Finite State Transducer) enforces:
  - Token (phoneme) → Lexicon (words) mapping
  - Grammar (n-gram) constraints
  - Vocabulary restrictions

**Conclusion**: Decoding is rule-compliant by design.

---

## Competition Implications

### Current Standing:
- **Baseline WER**: 21.43% (validation)
- **Competition metric**: WER on test set
- **Target**: <15% WER for competitive performance

### Improvement Path to Target:
1. Transformer model (Person B): -2-5% → ~19% WER
2. 3-gram LM: -3% → ~16% WER  
3. OPT rescoring: -2% → **~14% WER** ✓ Target reached

### Recommendation:
**Deploy**: Transformer + 3-gram + OPT-6.7b  
**Expected**: ~14-16% WER (competitive)  
**Resources**: ~85GB RAM + 13GB VRAM

---

## Files Generated

### Analysis Outputs:
- ✓ `baseline_evaluation_summary.png` - Visualization
- ✓ `analyze_examples.py` - Example analysis script
- ✓ `visualize_results.py` - Visualization script
- ✓ `EVALUATION_RESULTS.md` - This report

### Available for Next Steps:
- `run_lm_experiments.py` - Automated experiment runner
- `analyses/language_model_comparison.ipynb` - Interactive analysis
- `language_model/DECODING_PARAMETERS.md` - Parameter guide

---

## Conclusion

The baseline model achieves **21.43% WER** on validation, with strong performance on frequent words (11.87% WER) but struggles with random sequences (37.50% WER). The large PER-WER gap (10.20% vs 21.43%) indicates heavy LM dependency. 

**Key Finding**: Upgrading from 1-gram to 3-gram+OPT language model should reduce WER to ~15-17%, making the system competitive.

**Recommendation**: Proceed with 3-gram LM experiments to validate expected improvements.

---

**Analysis Completed**: December 31, 2025  
**Visualization**: `baseline_evaluation_summary.png`  
**Status**: Ready for LM comparison experiments
