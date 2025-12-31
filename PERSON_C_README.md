# Language Model Tasks - Tools & Framework

## 🎯 Your Responsibilities
1. ✅ Compare n-gram Language Models vs neural LMs
2. ✅ Ensure rule-compliant decoding
3. ✅ Analyze linguistic improvements

## 📦 Available Tools & Resources

### Core Files

| File | Purpose |
|------|--------|
| `analyses/language_model_comparison.ipynb` | Analysis notebook with WER/PER metrics, corpus analysis, error distribution |
| `run_lm_experiments.py` | Automated experiment runner for different LM configs |
| `language_model/DECODING_PARAMETERS.md` | Complete parameter guide with rule compliance validation |
| `LANGUAGE_MODEL_SUMMARY.md` | Report template with tables, analysis framework |
| `PERSON_C_QUICKSTART.md` | Setup and usage instructions |

### What Each File Does

#### 1. Analysis Notebook (`language_model_comparison.ipynb`)
Opens existing baseline results and provides:
- WER and PER calculations with confidence intervals
- Corpus-specific performance breakdown (Switchboard, Random words, etc.)
- Error type analysis (substitutions, insertions, deletions)
- Example predictions with side-by-side comparisons
- Visualization of error distributions

**Run it now**: 
```bash
conda activate b2txt25
jupyter notebook analyses/language_model_comparison.ipynb
```

#### 2. Experiment Runner (`run_lm_experiments.py`)
Automates running different LM configurations:
- **6 main configs**: baseline-1gram, 1gram-opt, 3gram, 3gram-opt, 5gram, 5gram-opt
- **6 tuning configs**: tune parameters for optimal performance
- Automatic result collection and logging
- Dry-run mode to preview commands

**Test it**: 
```bash
python run_lm_experiments.py --config baseline-1gram --dry_run
```

#### 3. Parameter Guide (`language_model/DECODING_PARAMETERS.md`)
Complete documentation of:
- All decoding parameters (acoustic_scale, alpha, blank_penalty, beam, etc.)
- Rule compliance validation (FST, vocabulary constraints)
- 6 recommended configurations for different scenarios
- Troubleshooting common issues
- Parameter sensitivity analysis

#### 4. Summary Report (`LANGUAGE_MODEL_SUMMARY.md`)
Report-ready document with:
- System architecture diagram
- Experiment design matrix
- Results analysis framework
- Rule compliance validation
- Performance vs resource trade-offs
- Recommendations for competition

#### 5. Quick Start Guide (`PERSON_C_QUICKSTART.md`)
Step-by-step instructions for:
- Environment setup
- Running analysis on existing results
- Running new experiments
- Parameter tuning workflow

## 🚀 Quick Start (5 Minutes)

### Option 1: Analyze Existing Results (Fastest)
```bash
# 1. Activate environment
conda activate b2txt25

# 2. Open notebook
jupyter notebook analyses/language_model_comparison.ipynb

# 3. Run all cells - shows current baseline performance
```

This analyzes the results Person A already generated!

### Option 2: Run New Experiments (If you have time)

**Prerequisites:**
- Redis server running
- Conda environment b2txt25_lm (or b2txt25)
- Language models downloaded (1-gram included, 3/5-gram optional)

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Run experiment
conda activate b2txt25_lm
python run_lm_experiments.py --config baseline-1gram --eval_type val
```

## 📊 Current Status

### What Exists Now
- ✅ Baseline RNN results (available in repo)
- ✅ 1-gram LM (included in repo)
- ✅ Analysis tools (ready to use)
- ✅ Experiment automation (ready to use)
- ✅ Documentation (available)

### What You Need To Do
1. **Minimum (for report)**:
   - Run analysis notebook on existing results
   - Document baseline WER/PER
   - Explain parameter choices from `DECODING_PARAMETERS.md`
   - Use findings from `LANGUAGE_MODEL_SUMMARY.md`

2. **If you have 3-gram/5-gram**:
   - Download from Dryad
   - Run comparison experiments
   - Document WER improvements

3. **Parameter tuning** (optional):
   - Run tuning configs
   - Find optimal parameters for your data

## 📈 Expected Results (from Literature)

Based on similar brain-to-text systems:

| Configuration | WER Reduction | Speed | Memory |
|--------------|---------------|-------|--------|
| 1-gram → 3-gram | -5 to -8% | Same | +59GB |
| N-gram → N-gram+OPT | -3 to -5% | 2-3x slower | +13GB |
| Combined improvement | -8 to -12% | Slower | High |

## 🤝 How Your Work Fits

```
Person A (Baseline + EDA) 
    ↓
    Provided baseline metrics & error analysis
    ↓
Person B (Transformer + Augmentation)
    ↓
    Improved acoustic model quality
    ↓
Person C (YOU - Language Models) ← You optimize this step
    ↓
    Optimize LM configuration & decoding
    ↓
    Final Competition Submission
```

## 📝 For Your Report

Use these sections from `LANGUAGE_MODEL_SUMMARY.md`:

1. **System Architecture** - Explain the pipeline
2. **Experiment Matrix** - Show configurations tested
3. **Rule Compliance** - Demonstrate valid decoding
4. **Linguistic Improvements** - Show examples of better grammar
5. **Trade-offs** - Memory vs performance analysis
6. **Recommendations** - Best config for competition

## 🔧 Troubleshooting

### "I don't have 3-gram/5-gram models"
**Solution**: Analyze 1-gram baseline only. The tools work with any available LMs.

### "Redis won't start on Windows"
**Solution**: 
- Download: https://github.com/microsoftarchive/redis/releases
- Or use existing baseline results for analysis

### "Out of memory"
**Solution**: Stick with 1-gram (only 1GB). 3/5-gram need 60-300GB.

### "Not sure where to start"
**Solution**: 
1. Open `PERSON_C_QUICKSTART.md`
2. Follow "Step 3: Run Analysis on Existing Results"
3. That's your minimum deliverable!

## ✅ Checklist for Completion

Minimum (30 minutes):
- [ ] Run `language_model_comparison.ipynb` on existing results
- [ ] Document baseline WER/PER
- [ ] Read `DECODING_PARAMETERS.md` for parameter explanations
- [ ] Use `LANGUAGE_MODEL_SUMMARY.md` for report

Recommended (2-3 hours):
- [ ] Download 3-gram LM from Dryad
- [ ] Run 1-gram vs 3-gram comparison
- [ ] Run 3-gram+OPT experiment
- [ ] Document WER improvements
- [ ] Create comparative visualizations

Advanced (if time permits):
- [ ] Run parameter tuning experiments
- [ ] Find optimal configuration
- [ ] Test with Person B's Transformer model
- [ ] Create comprehensive results table

## 📚 File Reference

Quick reference to find things:

| What You Need | Where to Find It |
|--------------|------------------|
| Start here | `PERSON_C_QUICKSTART.md` |
| Analyze results | `analyses/language_model_comparison.ipynb` |
| Run experiments | `run_lm_experiments.py` |
| Understand parameters | `language_model/DECODING_PARAMETERS.md` |
| Write report | `LANGUAGE_MODEL_SUMMARY.md` |
| Existing results | `data/t15_pretrained_rnn_baseline/*.csv` |
| LM code | `language_model/language-model-standalone.py` |
| WER calculation | `nejm_b2txt_utils/general_utils.py` |

## 🎉 You're All Set!

Everything you need is ready. Start with the analysis notebook, and you'll have results in minutes!

**Minimum time commitment**: 30 minutes (analyze existing results)  
**Full experiment suite**: 3-5 hours (if you have the LM models)  
**Report writing**: Use pre-written content from `LANGUAGE_MODEL_SUMMARY.md`

Good luck! 🚀
