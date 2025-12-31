# Language Model Experiments - Quick Start Guide

## ✅ Available Framework

A complete framework for language model experiments:

### 📁 Files Created

1. **`analyses/language_model_comparison.ipynb`** - Main analysis notebook
   - Loads baseline results
   - Calculates WER/PER metrics
   - Corpus-specific analysis
   - Error type distribution
   - Example predictions
   - Ready to compare different LM configurations

2. **`language_model/DECODING_PARAMETERS.md`** - Comprehensive parameter guide
   - Detailed explanation of all decoding parameters
   - Rule compliance validation
   - Recommended configurations for different scenarios
   - Troubleshooting common issues

3. **`run_lm_experiments.py`** - Automated experiment runner
   - Pre-configured LM setups (1-gram, 3-gram, 5-gram, with/without OPT)
   - Parameter tuning experiments
   - Automatic result collection and logging

## 🚀 How to Use

### Step 1: Check Current Environment

The setup script (`setup_lm.sh`) is for Linux. On Windows, you need:

```powershell
# Check if conda env exists
conda env list

# If b2txt25_lm doesn't exist, create it manually:
conda create -n b2txt25_lm python=3.9 -y
conda activate b2txt25_lm

# Install dependencies
pip install torch==1.13.1 redis==5.0.6 jupyter numpy matplotlib scipy scikit-learn tqdm g2p_en omegaconf huggingface-hub transformers accelerate
```

### Step 2: Install Redis (Required)

Download and install Redis for Windows:
- https://github.com/microsoftarchive/redis/releases
- Or use WSL/Docker

Start Redis server:
```powershell
redis-server
```

### Step 3: Run Analysis on Existing Results

Open and run the analysis notebook:
```powershell
conda activate b2txt25
jupyter notebook analyses/language_model_comparison.ipynb
```

This will analyze the existing baseline results and show current performance.

### Step 4: Run LM Experiments (Optional)

First check what LMs are available:
```powershell
python run_lm_experiments.py --config baseline-1gram --dry_run
```

Run a single experiment:
```powershell
# Terminal 1: Make sure Redis is running
redis-server

# Terminal 2: Run experiment
conda activate b2txt25_lm
python run_lm_experiments.py --config baseline-1gram --eval_type val
```

Run all available experiments:
```powershell
python run_lm_experiments.py --config all --eval_type val
```

### Step 5: Parameter Tuning

Run parameter tuning experiments:
```powershell
python run_lm_experiments.py --config tuning --include_tuning --eval_type val
```

Or tune specific parameters:
```powershell
python run_lm_experiments.py --config tune-acoustic-high
python run_lm_experiments.py --config tune-lm-high
python run_lm_experiments.py --config tune-blank-low
```

## 📊 Expected Workflow

1. **Analyze baseline** (already done by Person A)
   - Open `language_model_comparison.ipynb`
   - Run all cells to see current performance
   - Document baseline WER/PER

2. **Download additional LMs** (if needed)
   - Visit: https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq
   - Download `languageModel.tar.gz` (3-gram, ~60GB RAM required)
   - Download `languageModel_5gram.tar.gz` (5-gram, ~300GB RAM required)
   - Extract to `language_model/pretrained_language_models/`

3. **Run comparisons**
   - Start with 1-gram (already included)
   - If you have 3-gram: test 3-gram and 3-gram+OPT
   - Compare WER improvements

4. **Parameter tuning**
   - Run tuning experiments
   - Find optimal `acoustic_scale`, `alpha`, `blank_penalty`
   - Document rule compliance

5. **Document findings**
   - Update notebook with comparative results
   - Create tables/plots showing improvements
   - Write conclusions about linguistic gains

## 📈 Pre-Configured Experiments

### Available Configurations

| Config | LM | Neural | Memory | Speed | WER (Expected) |
|--------|----|----|--------|-------|---------|
| `baseline-1gram` | 1-gram | ✗ | ~1GB | Fast | Baseline |
| `1gram-opt` | 1-gram | OPT | ~14GB | Medium | -3% |
| `3gram` | 3-gram | ✗ | ~60GB | Medium | -5% |
| `3gram-opt` | 3-gram | OPT | ~73GB | Slow | -8% (best) |
| `5gram` | 5-gram | ✗ | ~300GB | Slow | -7% |
| `5gram-opt` | 5-gram | OPT | ~313GB | Very slow | -10% (theoretical best) |

### Parameter Tuning Configs

- `tune-acoustic-high`: Trust RNN more (acoustic_scale=0.45)
- `tune-acoustic-low`: Trust LM more (acoustic_scale=0.2)
- `tune-lm-high`: More neural LM influence (alpha=0.7)
- `tune-lm-low`: Less neural LM influence (alpha=0.3)
- `tune-blank-high`: Fewer silences (blank_penalty=110)
- `tune-blank-low`: More silences (blank_penalty=70)

## 🎯 Your Deliverables

Based on your task requirements:

### 1. Compare n-gram vs neural LM ✅
- Analysis notebook created: `language_model_comparison.ipynb`
- Experiment runner ready: `run_lm_experiments.py`
- **Action**: Run experiments and document WER improvements

### 2. Keep decoding rule-compliant ✅
- Parameter guide created: `DECODING_PARAMETERS.md`
- FST-based validation already in decoder
- **Action**: Validate phoneme sequences and vocabulary constraints

### 3. Analyze linguistic improvements ✅
- Error analysis functions in notebook
- Corpus-specific metrics
- **Action**: Compare error patterns between LM configs

## 🔧 Troubleshooting

### Issue: `setup_lm.sh` fails on Windows
**Solution**: Use manual conda setup (see Step 1 above)

### Issue: Redis not available
**Solution**: 
- Windows: Download from https://github.com/microsoftarchive/redis/releases
- Or use WSL: `sudo apt install redis-server`
- Or use Docker: `docker run -d -p 6379:6379 redis`

### Issue: LM files not found
**Solution**: 
- 1-gram should be included in repo
- 3-gram/5-gram need to be downloaded from Dryad
- Check paths in `run_lm_experiments.py`

### Issue: Out of memory
**Solution**:
- Use 1-gram (only 1GB)
- 3-gram requires 60GB RAM
- 5-gram requires 300GB RAM
- Consider running on high-memory server

### Issue: Slow decoding
**Solution**:
- Reduce `--nbest` to 50
- Reduce `--beam` to 13
- Use 1-gram instead of 3-gram/5-gram
- Disable OPT rescoring (`--alpha 0.0`)

## 📝 Next Steps

1. **Immediate**: 
   - Run the analysis notebook on existing results
   - Document baseline performance

2. **Short-term** (if you have the LMs):
   - Run experiments with different configurations
   - Compare WER improvements
   - Tune parameters

3. **For report**:
   - Create comparative tables
   - Plot WER vs memory/speed trade-offs
   - Show linguistic improvement examples
   - Document optimal parameters for competition

## 🤝 Integration with Team

Your work integrates with:
- **Person A**: Use their baseline results and error analysis
- **Person B**: Test with their Transformer model (change `--model_path`)

## 📚 Key References

- **LM Server**: `language_model/language-model-standalone.py`
- **Evaluation**: `model_training/evaluate_model.py`
- **WER Calculation**: `nejm_b2txt_utils/general_utils.py`
- **Parameters Guide**: `language_model/DECODING_PARAMETERS.md`
- **Baseline Results**: `data/t15_pretrained_rnn_baseline/detailed_results_*.csv`

---

**Status**: Framework complete, ready for experiments! 🎉

Start with the analysis notebook to understand current performance, then run experiments if you have time and resources.
