import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import collections
import math

# Import our custom modules
from src.models.experts import Expert
from src.metrics.calibration import TemperatureScaler
from src.data.dataloader_utils import get_expert_training_dataloaders

# --- TLC-STYLE LOSS FUNCTION ---
class TLCExpertLoss(nn.Module):
    """
    TLC-style loss adapted for single expert training (not multi-expert).
    Incorporates evidential learning, margin adjustment, and diversity regularization.
    """
    def __init__(self, cls_num_list, max_m=0.5, reweight_epoch=-1, reweight_factor=0.05, 
                 annealing=500, diversity_weight=0.01, kl_weight_schedule='linear'):
        super(TLCExpertLoss, self).__init__()
        self.reweight_epoch = reweight_epoch
        self.annealing = annealing
        self.diversity_weight = diversity_weight
        self.kl_weight_schedule = kl_weight_schedule
        
        # Margin computation: inversely related to class frequency
        m_list = 1. / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
        
        # Effective number reweighting (for later epochs)
        if reweight_epoch != -1:
            idx = 1
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights_enabled = None
            
        # Diversity reweighting
        cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
        C = len(cls_num_list)
        per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor
        per_cls_weights = per_cls_weights / np.max(per_cls_weights)
        self.per_cls_weights_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        
        # KL annealing schedule
        self.T = (reweight_epoch + annealing) / reweight_factor if reweight_epoch != -1 else annealing
        
    def to(self, device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)
        if self.per_cls_weights_diversity is not None:
            self.per_cls_weights_diversity = self.per_cls_weights_diversity.to(device)
        return self
        
    def _hook_before_epoch(self, epoch):
        """Update epoch-dependent parameters."""
        self.current_epoch = epoch
        if self.reweight_epoch != -1:
            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.use_diversity_weights = True
            else:
                self.per_cls_weights_base = None
                self.use_diversity_weights = False
        else:
            self.per_cls_weights_base = None
            self.use_diversity_weights = False
            
    def get_final_output(self, x, y):
        """Apply margin adjustment to ground-truth class logits."""
        index = torch.zeros_like(x, dtype=torch.bool, device=x.device)
        index.scatter_(1, y.data.view(-1, 1), 1)
        index_float = index.float()
        
        # Compute per-batch margins
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        
        # Apply margin to ground-truth class (reduced from 30 to 10 for stability)
        x_m = x - 10 * batch_m
        
        # NUMERICAL STABILIZATION: Clamp logits before exp to prevent overflow/underflow
        adjusted_logits = torch.where(index, x_m, x)
        adjusted_logits = torch.clamp(adjusted_logits, min=-50, max=50)  # Prevent extreme values
        
        # Convert to concentration parameters (evidential) with minimum value
        alpha = torch.exp(adjusted_logits) + 1e-6  # Add small epsilon to prevent alpha=0
        return alpha
        
    def forward(self, logits, targets, epoch, global_mean_logits=None):
        """
        TLC-style loss for single expert.
        
        Args:
            logits: [B, C] raw logits from expert
            targets: [B] ground truth labels  
            epoch: current training epoch
            global_mean_logits: [B, C] mean logits across all samples (for diversity)
        """
        # 1. Evidential learning: convert logits to Dirichlet concentration
        alpha = self.get_final_output(logits, targets)  # [B, C]
        S = alpha.sum(dim=1, keepdim=True)  # [B, 1]
        
        # 2. Evidential NLL loss with numerical stabilization
        # Clamp alpha and S to prevent log(0) and ensure numerical stability
        alpha_clamped = torch.clamp(alpha, min=1e-8, max=1e8)
        S_clamped = torch.clamp(S, min=1e-8, max=1e8)
        
        # Compute log probabilities safely
        log_alpha = torch.log(alpha_clamped)
        log_S = torch.log(S_clamped)
        log_probs = log_alpha - log_S
        
        # Check for NaN/Inf in log probabilities
        log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.full_like(log_probs, -100.0))
        
        nll_loss = F.nll_loss(
            log_probs, 
            targets, 
            weight=self.per_cls_weights_base, 
            reduction='none'
        )  # [B]
        
        # Safety check for NaN in loss
        nll_loss = torch.where(torch.isfinite(nll_loss), nll_loss, torch.zeros_like(nll_loss))
        
        # 3. KL regularization (encourage concentration on true class)
        if self.kl_weight_schedule == 'linear':
            kl_weight = epoch / self.T
        elif self.kl_weight_schedule == 'exponential':
            kl_weight = 1.0 - math.exp(-epoch / (self.T / 5))
        else:
            kl_weight = 1.0
            
        if kl_weight > 0:
            # One-hot encoding for true class
            yi = F.one_hot(targets, num_classes=alpha.shape[1]).float()  # [B, C]
            
            # Adjusted Dirichlet parameters: α̃ = y + (1-y)(α+1)
            alpha_tilde = yi + (1 - yi) * (alpha + 1)  # [B, C]
            
            # NUMERICAL STABILIZATION: Clamp alpha_tilde to prevent NaN in gamma functions
            alpha_tilde = torch.clamp(alpha_tilde, min=1e-6, max=1e6)
            S_tilde = alpha_tilde.sum(dim=1, keepdim=True)  # [B, 1]
            
            # SAFE KL divergence computation with checks for NaN/Inf
            try:
                # KL divergence: KL(Dir(α̃) || Uniform) with numerical stability
                num_classes_tensor = torch.tensor(alpha_tilde.shape[1], dtype=torch.float, device=alpha.device)
                
                # Compute each term separately with clamping
                term1 = torch.lgamma(torch.clamp(S_tilde, min=1e-6, max=1e6))
                term2 = torch.lgamma(torch.clamp(num_classes_tensor, min=1e-6, max=1e6))
                term3 = torch.lgamma(torch.clamp(alpha_tilde, min=1e-6, max=1e6)).sum(dim=1, keepdim=True)
                
                # Digamma terms with clamping
                digamma_alpha = torch.digamma(torch.clamp(alpha_tilde, min=1e-6, max=1e6))
                digamma_S = torch.digamma(torch.clamp(S_tilde, min=1e-6, max=1e6))
                term4 = ((alpha_tilde - 1) * (digamma_alpha - digamma_S)).sum(dim=1, keepdim=True)
                
                kl_div = (term1 - term2 - term3 + term4).squeeze(-1)  # [B]
                
                # Check for NaN/Inf and replace with zeros if found
                kl_div = torch.where(torch.isfinite(kl_div), kl_div, torch.zeros_like(kl_div))
                
                # Additional safety: clamp the final KL divergence
                kl_div = torch.clamp(kl_div, min=-100, max=100)
                
            except Exception as e:
                print(f"Warning: KL computation failed with error {e}, skipping KL term")
                kl_div = torch.zeros_like(nll_loss)
            
            nll_loss = nll_loss + kl_weight * kl_div
            
        # 4. Diversity regularization (prevent mode collapse)
        diversity_loss = 0.0
        if self.diversity_weight > 0 and global_mean_logits is not None:
            if self.use_diversity_weights:
                diversity_temperature = self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = 1.0
                temperature_mean = 1.0
                
            # Ensure global_mean_logits is on the same device as logits
            if global_mean_logits.device != logits.device:
                global_mean_logits = global_mean_logits.to(logits.device)
                
            # KL divergence from mean prediction (encourage diversity)
            output_dist = F.log_softmax(logits / diversity_temperature, dim=1)
            with torch.no_grad():
                mean_output_dist = F.softmax(global_mean_logits / diversity_temperature, dim=1)
            diversity_loss = F.kl_div(output_dist, mean_output_dist, reduction='none').sum(dim=1)  # [B]
            diversity_loss = -self.diversity_weight * temperature_mean * temperature_mean * diversity_loss
            
            nll_loss = nll_loss + diversity_loss
            
        # Final safety check: ensure the final loss is finite
        final_loss = nll_loss.mean()
        if not torch.isfinite(final_loss):
            print(f"Warning: Non-finite loss detected at epoch {epoch}, returning zero loss")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return final_loss

# --- TLC EXPERT CONFIGURATIONS ---
TLC_EXPERT_CONFIGS = {
    'tlc_ce': {
        'name': 'tlc_ce_expert',
        'loss_params': {
            'max_m': 0.3,  # Smaller margin for CE-style expert
            'reweight_epoch': -1,  # No reweighting
            'reweight_factor': 0.05,
            'annealing': 300,
            'diversity_weight': 0.005,  # Light diversity
        },
        'epochs': 200,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'dropout_rate': 0.1,
        'milestones': [160, 180],
        'gamma': 0.01,
        'warmup_epochs': 5,
    },
    'tlc_balanced': {
        'name': 'tlc_balanced_expert', 
        'loss_params': {
            'max_m': 0.5,  # Standard margin
            'reweight_epoch': 160,  # Enable reweighting after 160 epochs
            'reweight_factor': 0.05,
            'annealing': 500,
            'diversity_weight': 0.01,
        },
        'epochs': 200,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'dropout_rate': 0.1,
        'milestones': [160, 180],
        'gamma': 0.01,
        'warmup_epochs': 5,
    },
    'tlc_tail_focused': {
        'name': 'tlc_tail_expert',
        'loss_params': {
            'max_m': 0.7,  # Large margin for tail focus
            'reweight_epoch': 120,  # Earlier reweighting
            'reweight_factor': 0.08,  # Stronger reweighting
            'annealing': 400,
            'diversity_weight': 0.015,  # Higher diversity
        },
        'epochs': 200,
        'lr': 0.1,
        'weight_decay': 5e-4,
        'dropout_rate': 0.15,  # More dropout for regularization
        'milestones': [140, 170],
        'gamma': 0.01,
        'warmup_epochs': 5,
    }
}

# --- GLOBAL CONFIGURATION ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'data_root': './data',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
        'num_groups': 2,
    },
    'train_params': {
        'batch_size': 128,
        'momentum': 0.9,
        'nesterov': True,
    },
    'output': {
        'checkpoints_dir': './checkpoints/experts_tlc',
        'logits_dir': './outputs/logits_tlc',
    },
    'seed': 42
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HELPER FUNCTIONS ---

def get_class_counts():
    """Get CIFAR-100-LT class counts for imbalance factor 100."""
    # This should match your existing data preparation
    from src.data.datasets import get_cifar100_lt_counts
    return get_cifar100_lt_counts(imb_factor=100)

def get_dataloaders():
    """Get train and validation dataloaders."""
    print("Loading CIFAR-100-LT datasets...")
    
    train_loader, val_loader = get_expert_training_dataloaders(
        batch_size=CONFIG['train_params']['batch_size'],
        num_workers=4
    )
    
    print(f"  Train loader: {len(train_loader)} batches ({len(train_loader.dataset):,} samples)")
    print(f"  Val loader: {len(val_loader)} batches ({len(val_loader.dataset):,} samples)")
    
    return train_loader, val_loader

def validate_model(model, val_loader, device):
    """Validate model with group-wise metrics."""
    model.eval()
    correct = 0
    total = 0
    
    group_correct = {'head': 0, 'tail': 0}
    group_total = {'head': 0, 'tail': 0}
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Group-wise accuracy (Head: 0-49, Tail: 50-99)
            for i, target in enumerate(targets):
                pred = predicted[i]
                if target < 50:  # Head classes
                    group_total['head'] += 1
                    if pred == target:
                        group_correct['head'] += 1
                else:  # Tail classes
                    group_total['tail'] += 1
                    if pred == target:
                        group_correct['tail'] += 1
    
    overall_acc = 100 * correct / total
    
    group_accs = {}
    for group in ['head', 'tail']:
        if group_total[group] > 0:
            group_accs[group] = 100 * group_correct[group] / group_total[group]
        else:
            group_accs[group] = 0.0
    
    return overall_acc, group_accs

def export_logits_for_all_splits(model, expert_name):
    """Export logits for all dataset splits (same as original)."""
    print(f"Exporting logits for expert '{expert_name}'...")
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    splits_dir = Path(CONFIG['dataset']['splits_dir'])
    output_dir = Path(CONFIG['output']['logits_dir']) / CONFIG['dataset']['name'] / expert_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define splits to export
    splits_info = [
        {'name': 'train', 'dataset_type': 'train', 'file': 'train_indices.json'},
        {'name': 'tuneV', 'dataset_type': 'train', 'file': 'tuneV_indices.json'},
        {'name': 'val_small', 'dataset_type': 'train', 'file': 'val_small_indices.json'},
        {'name': 'val_lt', 'dataset_type': 'test', 'file': 'val_lt_indices.json'},
        {'name': 'test_lt', 'dataset_type': 'test', 'file': 'test_lt_indices.json'},
        {'name': 'calib', 'dataset_type': 'train', 'file': 'calib_indices.json'},
    ]
    
    for split_info in splits_info:
        split_name = split_info['name']
        dataset_type = split_info['dataset_type']
        indices_file = split_info['file']
        indices_path = splits_dir / indices_file
        
        if not indices_path.exists():
            print(f"  Warning: {indices_file} not found, skipping {split_name}")
            continue
            
        # Load appropriate base dataset
        if dataset_type == 'train':
            base_dataset = torchvision.datasets.CIFAR100(
                root=CONFIG['dataset']['data_root'], train=True, transform=transform)
        else:
            base_dataset = torchvision.datasets.CIFAR100(
                root=CONFIG['dataset']['data_root'], train=False, transform=transform)
        
        # Load indices and create subset
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        subset = Subset(base_dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=4)
        
        # Export logits
        all_logits = []
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"Exporting {split_name}"):
                logits = model.get_calibrated_logits(inputs.to(DEVICE))
                all_logits.append(logits.cpu())
        
        all_logits = torch.cat(all_logits)
        torch.save(all_logits.to(torch.float16), output_dir / f"{split_name}_logits.pt")
        print(f"  Exported {split_name}: {len(indices):,} samples")
    
    print(f"✅ All logits exported to: {output_dir}")

# --- CORE TRAINING FUNCTIONS ---

def train_single_tlc_expert(expert_key):
    """Train a single TLC-style expert."""
    if expert_key not in TLC_EXPERT_CONFIGS:
        raise ValueError(f"Expert '{expert_key}' not found in TLC_EXPERT_CONFIGS")
    
    expert_config = TLC_EXPERT_CONFIGS[expert_key]
    expert_name = expert_config['name']
    
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING TLC EXPERT: {expert_name.upper()}")
    print(f"🎯 Loss Type: TLC Evidential Learning")
    print(f"{'='*60}")
    
    # Setup
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    train_loader, val_loader = get_dataloaders()
    
    # Get class counts for loss function
    class_counts = get_class_counts()
    print(f"✅ Class counts loaded: {len(class_counts)} classes")
    
    # Model
    model = Expert(
        num_classes=CONFIG['dataset']['num_classes'],
        backbone_name='cifar_resnet32',
        dropout_rate=expert_config['dropout_rate'],
        init_weights=True
    ).to(DEVICE)
    
    # TLC Loss
    criterion = TLCExpertLoss(
        cls_num_list=class_counts,
        **expert_config['loss_params']
    ).to(DEVICE)
    
    print(f"✅ Loss Function: TLCExpertLoss")
    print(f"   - Max margin: {expert_config['loss_params']['max_m']}")
    print(f"   - Reweight epoch: {expert_config['loss_params']['reweight_epoch']}")
    print(f"   - Diversity weight: {expert_config['loss_params']['diversity_weight']}")
    
    # Print model summary
    print("📊 Model Architecture:")
    model.summary()
    
    # Optimizer with Nesterov momentum (like TLC)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=expert_config['lr'],
        momentum=CONFIG['train_params']['momentum'],
        weight_decay=expert_config['weight_decay'],
        nesterov=CONFIG['train_params']['nesterov']
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        # Warmup
        warmup_epochs = expert_config['warmup_epochs']
        if epoch < warmup_epochs:
            return float(1 + epoch) / warmup_epochs
        
        # Multi-step decay
        lr = 1.0
        for milestone in expert_config['milestones']:
            if epoch >= milestone:
                lr *= expert_config['gamma']
        return lr
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training setup
    best_acc = 0.0
    checkpoint_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"best_{expert_name}.pth"
    
    # Collect global mean logits for diversity (simple running average)
    global_mean_logits = None
    
    # Training loop
    for epoch in range(expert_config['epochs']):
        # Hook before epoch for loss function
        criterion._hook_before_epoch(epoch)
        
        # Train
        model.train()
        running_loss = 0.0
        epoch_logits = []
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{expert_config['epochs']}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Collect logits for diversity computation
            epoch_logits.append(outputs.detach().cpu())
            
            # Compute loss (with current global mean if available)
            # Move global_mean_logits to same device as outputs
            global_mean_on_device = global_mean_logits.to(DEVICE) if global_mean_logits is not None else None
            loss = criterion(outputs, targets, epoch, global_mean_on_device)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Update global mean logits (exponential moving average)
        current_mean = torch.cat(epoch_logits).mean(dim=0, keepdim=True)  # [1, C]
        if global_mean_logits is None:
            global_mean_logits = current_mean
        else:
            global_mean_logits = 0.9 * global_mean_logits + 0.1 * current_mean
        
        scheduler.step()
        
        # Validate
        val_acc, group_accs = validate_model(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1:3d}: Loss={running_loss/len(train_loader):.4f}, "
            f"Val Acc={val_acc:.2f}%, Head={group_accs['head']:.1f}%, "
            f"Tail={group_accs['tail']:.1f}%, LR={scheduler.get_last_lr()[0]:.5f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"💾 New best! Saving to {best_model_path}")
            torch.save(model.state_dict(), best_model_path)
    
    # Post-training: Calibration
    print(f"\n--- 🔧 POST-PROCESSING: {expert_name} ---")
    model.load_state_dict(torch.load(best_model_path))
    
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(model, val_loader, DEVICE)
    model.set_temperature(optimal_temp)
    print(f"✅ Temperature calibration: T = {optimal_temp:.3f}")
    
    final_model_path = checkpoint_dir / f"final_calibrated_{expert_name}.pth"
    torch.save(model.state_dict(), final_model_path)
    
    # Final validation
    final_acc, final_group_accs = validate_model(model, val_loader, DEVICE)
    print(f"📊 Final Results - Overall: {final_acc:.2f}%, "
        f"Head: {final_group_accs['head']:.1f}%, "
        f"Tail: {final_group_accs['tail']:.1f}%")
    
    # Export logits
    export_logits_for_all_splits(model, expert_name)
    
    print(f"✅ COMPLETED: {expert_name}")
    return final_model_path

def main():
    """Main training script - trains all 3 TLC-style experts."""
    print("🚀 AR-GSE TLC Expert Training Pipeline")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {CONFIG['dataset']['name']}")
    print(f"TLC experts to train: {list(TLC_EXPERT_CONFIGS.keys())}")
    
    results = {}
    
    for expert_key in TLC_EXPERT_CONFIGS.keys():
        try:
            model_path = train_single_tlc_expert(expert_key)
            results[expert_key] = {'status': 'success', 'path': model_path}
        except Exception as e:
            print(f"❌ Failed to train {expert_key}: {e}")
            results[expert_key] = {'status': 'failed', 'error': str(e)}
            continue
    
    print(f"\n{'='*60}")
    print("🏁 TLC TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for expert_key, result in results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"{status} {expert_key}: {result['status']}")
        if result['status'] == 'failed':
            print(f"    Error: {result['error']}")
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"\nSuccessfully trained {successful}/{len(TLC_EXPERT_CONFIGS)} TLC experts")

if __name__ == '__main__':
    main()
