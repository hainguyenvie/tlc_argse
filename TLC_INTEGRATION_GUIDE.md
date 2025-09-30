# TLC Integration Guide for AR-GSE Pipeline

## Overview

This guide explains how the **Trustworthy Long-Tailed Classification (TLC)** method has been integrated into your AR-GSE pipeline to improve expert training performance on imbalanced datasets.

## Key TLC Innovations Integrated

### 1. **Evidential Learning**
- Converts logits to Dirichlet concentration parameters α = exp(logits)
- Provides principled uncertainty quantification
- Better handles prediction confidence on tail classes

### 2. **Margin-Based Adjustment**
- Applies class-frequency-based margins to ground-truth logits
- Formula: `margin_c = max_m / sqrt(sqrt(count_c))`
- Larger margins for rarer classes (stronger regularization)

### 3. **KL Regularization**
- Encourages concentration on true class using Dirichlet KL divergence
- Annealed over training: `weight = epoch / total_epochs`
- Prevents overconfidence on uncertain predictions

### 4. **Diversity Regularization**
- Prevents expert collapse by encouraging diversity from global mean
- Uses KL divergence: `KL(expert_output || global_mean_output)`
- Weighted by class-frequency-based temperature scaling

### 5. **Adaptive Reweighting**
- Effective number reweighting activated after specified epoch
- Formula: `weight_c = (1 - β^count_c) / (1 - β)` where β=0.9999
- Balances contribution of head vs tail classes

## File Structure

```
src/train/
├── train_expert_tlc.py      # New TLC-style expert training
├── train_expert.py          # Original expert training (kept for comparison)
└── ...

run_tlc_pipeline.py          # Complete TLC-enhanced pipeline
TLC_INTEGRATION_GUIDE.md     # This guide
```

## Usage

### Option 1: Run Complete TLC Pipeline
```bash
python run_tlc_pipeline.py
```

This automatically runs all 4 steps with TLC experts:
1. `python -m src.train.train_expert_tlc`
2. `python run_tlc_gating.py --mode selective`  
3. `python run_tlc_plugin.py`
4. `python run_tlc_eval.py`

### Option 2: Manual Step-by-Step (for debugging)
```bash
# Step 1: Train TLC experts
python -m src.train.train_expert_tlc

# Step 2: Train gating with TLC expert logits
python -m src.train.train_gating_only --mode selective
# (Update CONFIG['experts']['logits_dir'] to './outputs/tlc_logit/')

# Step 3: Plugin optimization
python run_improved_eg_outer.py  
# (Update CONFIG to use TLC logits)

# Step 4: Evaluation
python -m src.train.eval_gse_plugin
# (Update CONFIG to use TLC checkpoint)
```

## TLC Expert Configurations

### `tlc_ce` - CE-style with light TLC enhancements
- **Purpose**: Baseline expert with minimal TLC modifications
- **Margin**: 0.3 (moderate)
- **Reweighting**: Disabled
- **Use case**: Head-class focused expert

### `tlc_balanced` - Standard TLC configuration
- **Purpose**: Balanced head-tail performance
- **Margin**: 0.5 (standard)
- **Reweighting**: After epoch 160
- **Use case**: General-purpose expert

### `tlc_tail_focused` - Aggressive tail optimization
- **Purpose**: Maximize tail class performance
- **Margin**: 0.7 (large)
- **Reweighting**: Earlier (epoch 120) and stronger
- **Use case**: Tail-specialized expert

## Expected Improvements

### 1. **Better Uncertainty Calibration**
- ECE (Expected Calibration Error) should decrease
- More reliable confidence estimates, especially on tail classes

### 2. **Improved Worst-Group Performance**
- Lower worst-group error in RC curves
- Better AURC (Area Under Risk-Coverage curve) scores

### 3. **Enhanced Expert Diversity**
- Reduced expert collapse via diversity regularization
- Better complementary behavior between experts

### 4. **Tail Class Performance**
- Higher accuracy on rare classes (50-99 in CIFAR-100-LT)
- Better precision/recall balance across all classes

## Comparison with Original Pipeline

| Aspect | Original | TLC-Enhanced |
|--------|----------|--------------|
| **Expert Training** | CE/LogitAdjust/BalancedSoftmax | Evidential learning + margins |
| **Loss Function** | Simple cross-entropy variants | Multi-component TLC loss |
| **Uncertainty** | Temperature scaling only | Principled Dirichlet uncertainty |
| **Class Balance** | Fixed reweighting schemes | Adaptive effective-number reweighting |
| **Expert Diversity** | Implicit via different losses | Explicit diversity regularization |
| **Tail Focus** | Limited by loss design | Margin-based + reweighting |

## Output Directories

- **Checkpoints**: `./checkpoints/experts_tlc/`
- **Logits**: `./outputs/tlc_logit/`
- **Plugin Results**: `./checkpoints/argse_tlc_improved/`
- **Final Metrics**: `./results_tlc_improved/`

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - TLC training uses more GPU memory due to evidential computations
   - Reduce batch size if needed in `CONFIG['train_params']['batch_size']`

2. **Slower Training**
   - TLC loss is more complex than simple CE
   - Each epoch takes ~20-30% longer but should converge better

3. **Logits Compatibility**
   - TLC experts output the same logit format as original experts
   - No changes needed in gating/plugin stages

### Performance Validation

Compare these metrics between original and TLC pipelines:

1. **AURC Balanced/Worst** (lower is better)
2. **ECE** (lower is better) 
3. **Plugin metrics at threshold** (coverage vs error trade-off)
4. **Group-wise accuracies** (head vs tail balance)

## Advanced Configuration

### Custom TLC Expert
To add a new TLC expert configuration:

```python
TLC_EXPERT_CONFIGS['custom_expert'] = {
    'name': 'custom_tlc_expert',
    'loss_params': {
        'max_m': 0.6,           # Margin strength
        'reweight_epoch': 140,   # When to enable reweighting
        'reweight_factor': 0.1,  # Reweighting strength  
        'annealing': 400,        # KL annealing schedule
        'diversity_weight': 0.02, # Diversity regularization
    },
    'epochs': 200,
    'lr': 0.1,
    'weight_decay': 5e-4,
    'dropout_rate': 0.12,
    'milestones': [150, 175],
    'gamma': 0.01,
    'warmup_epochs': 5,
}
```

### Loss Component Weights
Fine-tune the TLC loss components:

- **Margin adjustment**: Controls class-specific difficulty
- **KL weight**: Balances concentration vs diversity  
- **Diversity weight**: Prevents mode collapse
- **Reweight factor**: Tail class emphasis strength

## References

- **TLC Paper**: "Trustworthy Long-Tailed Classification" 
- **Evidential Learning**: "Evidential Deep Learning to Quantify Classification Uncertainty"
- **Effective Numbers**: "Class-Balanced Loss Based on Effective Number of Samples"
