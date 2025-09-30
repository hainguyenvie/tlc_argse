#!/usr/bin/env python3
"""
Complete AR-GSE pipeline using TLC-style expert training.
This script replaces the original 4-step pipeline with TLC-enhanced experts.
"""

import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with return code {e.returncode}")
        return False

def update_pipeline_configs():
    """Update configuration files to use TLC experts."""
    print("\n🔧 Updating pipeline configurations for TLC experts...")
    
    # Update expert names in gating training config
    gating_config_updates = {
        'experts': {
            'names': ['tlc_ce_expert', 'tlc_balanced_expert', 'tlc_tail_expert'],
            'logits_dir': './outputs/logits_tlc/',
        }
    }
    
    # Update plugin training config  
    plugin_config_updates = {
        'experts': {
            'names': ['tlc_ce_expert', 'tlc_balanced_expert', 'tlc_tail_expert'],
            'logits_dir': './outputs/logits_tlc',
        },
        'output': {
            'checkpoints_dir': './checkpoints/argse_tlc_improved/',
        }
    }
    
    # Update evaluation config
    eval_config_updates = {
        'experts': {
            'names': ['tlc_ce_expert', 'tlc_balanced_expert', 'tlc_tail_expert'],
            'logits_dir': './outputs/logits_tlc',
        },
        'plugin_checkpoint': './checkpoints/argse_tlc_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
        'output_dir': './results_tlc_improved/cifar100_lt_if100',
    }
    
    print("✅ Configuration updates prepared")
    return gating_config_updates, plugin_config_updates, eval_config_updates

def create_updated_scripts(gating_updates, plugin_updates, eval_updates):
    """Create updated scripts with TLC configurations."""
    
    # Create updated gating training script
    gating_script = f'''#!/usr/bin/env python3
"""
Updated gating training script for TLC experts.
"""
import sys
sys.path.append('.')

from src.train.train_gating_only import main, CONFIG

if __name__ == '__main__':
    print("🚀 TLC-Enhanced AR-GSE Gating Training")
    
    # Update config for TLC experts
    CONFIG['experts'].update({gating_updates['experts']})
    
    print(f"Using TLC experts: {{CONFIG['experts']['names']}}")
    print(f"Logits directory: {{CONFIG['experts']['logits_dir']}}")
    
    main()
'''
    
    # Create updated plugin training script  
    plugin_script = f'''#!/usr/bin/env python3
"""
Updated plugin training script for TLC experts.
"""
import sys
sys.path.append('.')

from src.train.gse_balanced_plugin import main, CONFIG

if __name__ == '__main__':
    print("🚀 TLC-Enhanced GSE Balanced Plugin Training")
    
    # Update config for TLC experts
    CONFIG['experts'].update({plugin_updates['experts']})
    CONFIG['output'].update({plugin_updates['output']})
    
    # Apply improvements
    CONFIG['plugin_params'].update({{
        'objective': 'balanced',
        'use_eg_outer': False,
        'eg_outer_T': 30,
        'eg_outer_xi': 0.2,
        'use_conditional_alpha': True,
        'M': 10,
        'alpha_steps': 4,
        'gamma': 0.25,
    }})
    
    print(f"Using TLC experts: {{CONFIG['experts']['names']}}")
    print(f"Output directory: {{CONFIG['output']['checkpoints_dir']}}")
    
    main()
'''
    
    # Create updated evaluation script
    eval_script = f'''#!/usr/bin/env python3
"""
Updated evaluation script for TLC experts.
"""
import sys
sys.path.append('.')

from src.train.eval_gse_plugin import main, CONFIG

if __name__ == '__main__':
    print("🚀 TLC-Enhanced GSE Plugin Evaluation")
    
    # Update config for TLC experts
    CONFIG['experts'].update({eval_updates['experts']})
    CONFIG.update({{
        'plugin_checkpoint': '{eval_updates['plugin_checkpoint']}',
        'output_dir': '{eval_updates['output_dir']}'
    }})
    
    print(f"Using TLC experts: {{CONFIG['experts']['names']}}")
    print(f"Plugin checkpoint: {{CONFIG['plugin_checkpoint']}}")
    print(f"Output directory: {{CONFIG['output_dir']}}")
    
    main()
'''
    
    # Write scripts to files
    Path('run_tlc_gating.py').write_text(gating_script)
    Path('run_tlc_plugin.py').write_text(plugin_script)
    Path('run_tlc_eval.py').write_text(eval_script)
    
    print("✅ Updated pipeline scripts created:")
    print("   - run_tlc_gating.py")
    print("   - run_tlc_plugin.py") 
    print("   - run_tlc_eval.py")

def main():
    """Run the complete TLC-enhanced AR-GSE pipeline."""
    print("🎯 TLC-Enhanced AR-GSE Pipeline")
    print("=" * 60)
    print("This pipeline uses Trustworthy Long-Tailed Classification")
    print("for expert training with evidential learning and uncertainty.")
    print("=" * 60)
    
    # Prepare configurations
    gating_updates, plugin_updates, eval_updates = update_pipeline_configs()
    create_updated_scripts(gating_updates, plugin_updates, eval_updates)
    
    # Pipeline steps
    steps = [
        {
            'cmd': [sys.executable, '-m', 'src.train.train_expert_tlc'],
            'desc': 'Step 1: Training TLC-style Experts',
            'required_outputs': [
                './outputs/logits_tlc/cifar100_lt_if100/tlc_ce_expert/',
                './outputs/logits_tlc/cifar100_lt_if100/tlc_balanced_expert/', 
                './outputs/logits_tlc/cifar100_lt_if100/tlc_tail_expert/',
            ]
        },
        {
            'cmd': [sys.executable, 'run_tlc_gating.py', '--mode', 'selective'],
            'desc': 'Step 2: Training AR-GSE Gating Network (Selective Mode)',
            'required_outputs': [
                './checkpoints/gating_pretrained/cifar100_lt_if100/gating_selective.ckpt'
            ]
        },
        {
            'cmd': [sys.executable, 'run_tlc_plugin.py'],
            'desc': 'Step 3: GSE Balanced Plugin Optimization',
            'required_outputs': [
                './checkpoints/argse_tlc_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt'
            ]
        },
        {
            'cmd': [sys.executable, 'run_tlc_eval.py'],
            'desc': 'Step 4: Final Evaluation and Metrics',
            'required_outputs': [
                './results_tlc_improved/cifar100_lt_if100/metrics.json',
                './results_tlc_improved/cifar100_lt_if100/rc_curve_comparison.png'
            ]
        }
    ]
    
    # Run pipeline
    success_count = 0
    for i, step in enumerate(steps, 1):
        success = run_command(step['cmd'], step['desc'])
        if success:
            success_count += 1
            
            # Check outputs exist
            missing_outputs = []
            for output_path in step['required_outputs']:
                if not Path(output_path).exists():
                    missing_outputs.append(output_path)
            
            if missing_outputs:
                print(f"⚠️  Warning: Expected outputs missing:")
                for path in missing_outputs:
                    print(f"   - {{path}}")
        else:
            print(f"💥 Pipeline failed at step {i}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TLC-ENHANCED PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Completed steps: {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        print("🎉 Complete pipeline executed successfully!")
        print("\nKey improvements from TLC integration:")
        print("✅ Evidential learning with uncertainty quantification")
        print("✅ Margin-based adjustment for class imbalance")
        print("✅ KL regularization for better concentration")
        print("✅ Diversity regularization to prevent mode collapse")
        print("✅ Adaptive reweighting based on effective numbers")
        
        print(f"\nResults available in: ./results_tlc_improved/cifar100_lt_if100/")
        print("Next steps:")
        print("- Compare metrics.json with original pipeline results")
        print("- Analyze RC curves for improvements in worst-group performance")
        print("- Check uncertainty calibration via ECE scores")
    else:
        print("❌ Pipeline incomplete. Check error messages above.")
        print("\nTroubleshooting tips:")
        print("- Ensure all required data splits exist in ./data/cifar100_lt_if100_splits/")
        print("- Check GPU memory availability for TLC training")
        print("- Verify all dependencies are installed")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
