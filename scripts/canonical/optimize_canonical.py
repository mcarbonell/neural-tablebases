"""
Optimize hyperparameters for canonical datasets.
"""
import subprocess
import sys
import json
import re
from datetime import datetime

def train_with_config(dataset_path, endgame, config):
    """Train with specific configuration."""
    print(f"\nTraining {endgame} with config: {config['name']}")
    
    cmd = [
        sys.executable, "src/train.py",
        "--data_path", dataset_path,
        "--epochs", str(config['epochs']),
        "--batch_size", str(config['batch_size']),
        "--lr", str(config['lr']),
        "--model", "mlp",
        "--patience", str(config.get('patience', 100))
    ]
    
    if config.get('hard_mining', True):
        cmd.append("--hard_mining")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Parse results
            accuracy_match = re.search(r'Best validation accuracy: (\d+\.\d+)', output)
            accuracy = float(accuracy_match.group(1)) * 100 if accuracy_match else None
            
            dtz_match = re.search(r'Best validation DTZ MAE: (\d+\.\d+)', output)
            dtz_mae = float(dtz_match.group(1)) if dtz_match else None
            
            # Find training time
            time_match = re.search(r'Total training time: (\d+\.\d+)s', output)
            training_time = float(time_match.group(1)) if time_match else None
            
            print(f"  Results: {accuracy:.2f}% accuracy, {dtz_mae:.2f} DTZ MAE")
            
            return {
                'success': True,
                'accuracy': accuracy,
                'dtz_mae': dtz_mae,
                'training_time': training_time,
                'config': config
            }
        else:
            print(f"  Training failed!")
            return {
                'success': False,
                'error': result.stderr[:500]
            }
            
    except subprocess.TimeoutExpired:
        print(f"  Training timed out")
        return {
            'success': False,
            'error': 'Timeout'
        }

def optimize_kpvk():
    """Optimize KPvK which had the largest accuracy drop."""
    print("="*60)
    print("OPTIMIZING KPvK CANONICAL (largest accuracy drop)")
    print("="*60)
    
    dataset_path = "data/KPvK_canonical.npz"
    original_accuracy = 99.88  # Original KPvK accuracy
    original_dtz = 0.06
    
    # Test configurations
    configs = [
        # Baseline (what we already tried)
        {
            'name': 'baseline',
            'epochs': 50,
            'batch_size': 1024,
            'lr': 0.001,
            'hard_mining': True
        },
        # More epochs
        {
            'name': 'more_epochs',
            'epochs': 200,
            'batch_size': 1024,
            'lr': 0.001,
            'hard_mining': True
        },
        # Smaller batch size (better for small dataset)
        {
            'name': 'small_batch',
            'epochs': 100,
            'batch_size': 512,
            'lr': 0.001,
            'hard_mining': True
        },
        # Lower learning rate
        {
            'name': 'low_lr',
            'epochs': 100,
            'batch_size': 1024,
            'lr': 0.0005,
            'hard_mining': True
        },
        # Higher learning rate
        {
            'name': 'high_lr',
            'epochs': 100,
            'batch_size': 1024,
            'lr': 0.002,
            'hard_mining': True
        },
        # More patience (longer training)
        {
            'name': 'more_patience',
            'epochs': 300,
            'batch_size': 1024,
            'lr': 0.001,
            'patience': 200,
            'hard_mining': True
        }
    ]
    
    results = {}
    
    for config in configs:
        result = train_with_config(dataset_path, "KPvK", config)
        results[config['name']] = result
        
        if result['success']:
            accuracy_diff = result['accuracy'] - original_accuracy
            print(f"  Δ Accuracy vs original: {accuracy_diff:+.2f}%")
    
    return results

def optimize_all():
    """Optimize all canonical datasets."""
    print("="*60)
    print("OPTIMIZING ALL CANONICAL DATASETS")
    print("="*60)
    
    datasets = {
        'KQvK': ('data/KQvK_canonical.npz', 99.94, 0.64),
        'KRvK': ('data/KRvK_canonical.npz', 100.00, 1.00),
        'KPvK': ('data/KPvK_canonical.npz', 99.88, 0.06)
    }
    
    # Best configuration from KPvK optimization
    best_config = {
        'name': 'optimized',
        'epochs': 200,
        'batch_size': 512,
        'lr': 0.001,
        'patience': 150,
        'hard_mining': True
    }
    
    all_results = {}
    
    for endgame, (dataset_path, orig_acc, orig_dtz) in datasets.items():
        print(f"\n{'='*40}")
        print(f"Optimizing {endgame}")
        print(f"{'='*40}")
        
        result = train_with_config(dataset_path, endgame, best_config)
        
        if result['success']:
            accuracy_diff = result['accuracy'] - orig_acc
            dtz_diff = result['dtz_mae'] - orig_dtz
            
            print(f"\n  Original: {orig_acc:.2f}%, {orig_dtz:.2f} DTZ MAE")
            print(f"  Canonical: {result['accuracy']:.2f}%, {result['dtz_mae']:.2f} DTZ MAE")
            print(f"  Δ Accuracy: {accuracy_diff:+.2f}%")
            print(f"  Δ DTZ MAE: {dtz_diff:+.2f}")
        
        all_results[endgame] = {
            'original_accuracy': orig_acc,
            'original_dtz_mae': orig_dtz,
            'result': result
        }
    
    return all_results, best_config

def main():
    print("="*60)
    print("CANONICAL FORMS HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # First optimize KPvK (worst case)
    print("\nPhase 1: Optimizing KPvK (largest accuracy drop)")
    kpvk_results = optimize_kpvk()
    
    # Then apply best config to all
    print("\n" + "="*60)
    print("Phase 2: Applying optimized config to all datasets")
    all_results, best_config = optimize_all()
    
    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    
    results_file = f"results/canonical_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'kpvk_optimization': kpvk_results,
            'all_results': all_results,
            'best_config': best_config,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nBest configuration:")
    print(f"  Epochs: {best_config['epochs']}")
    print(f"  Batch size: {best_config['batch_size']}")
    print(f"  Learning rate: {best_config['lr']}")
    print(f"  Patience: {best_config.get('patience', 100)}")
    print(f"  Hard mining: {best_config.get('hard_mining', True)}")
    
    print(f"\nFinal results vs original:")
    
    total_accuracy_diff = 0
    count = 0
    
    for endgame, data in all_results.items():
        if data['result']['success']:
            orig_acc = data['original_accuracy']
            canon_acc = data['result']['accuracy']
            diff = canon_acc - orig_acc
            
            print(f"\n{endgame}:")
            print(f"  Original: {orig_acc:.2f}%")
            print(f"  Canonical: {canon_acc:.2f}%")
            print(f"  Δ Accuracy: {diff:+.2f}%")
            
            total_accuracy_diff += diff
            count += 1
    
    if count > 0:
        avg_diff = total_accuracy_diff / count
        print(f"\nAverage Δ Accuracy: {avg_diff:+.2f}%")
        
        if avg_diff >= -0.1:
            print(f"\n✓ EXCELLENT: Canonical forms match or exceed original accuracy!")
        elif avg_diff >= -0.5:
            print(f"\n✓ GOOD: Canonical forms perform similarly to original")
        else:
            print(f"\n⚠ NEEDS IMPROVEMENT: Canonical forms underperform")

if __name__ == "__main__":
    main()