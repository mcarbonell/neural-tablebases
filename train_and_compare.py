"""
Train models on canonical datasets and compare with original.
"""
import subprocess
import sys
import json
import re
from datetime import datetime

def run_training(dataset_path, endgame, epochs=100):
    """Run training and capture results."""
    print(f"\n{'='*60}")
    print(f"Training {endgame} (canonical)")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "src/train.py",
        "--data_path", dataset_path,
        "--epochs", str(epochs),
        "--batch_size", "1024",
        "--lr", "0.001",
        "--model", "mlp"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            # Parse results
            output = result.stdout
            
            # Find best validation accuracy
            accuracy_match = re.search(r'Best validation accuracy: (\d+\.\d+)', output)
            accuracy = float(accuracy_match.group(1)) * 100 if accuracy_match else None
            
            # Find best validation DTZ MAE
            dtz_match = re.search(r'Best validation DTZ MAE: (\d+\.\d+)', output)
            dtz_mae = float(dtz_match.group(1)) if dtz_match else None
            
            # Find final epoch results
            last_lines = '\n'.join(output.split('\n')[-20:])  # Last 20 lines
            final_epoch_match = re.search(r'Epoch \d+/\d+.*Val Acc: (\d+\.\d+).*Val DTZ MAE: (\d+\.\d+)', last_lines)
            
            if final_epoch_match:
                final_acc = float(final_epoch_match.group(1)) * 100
                final_dtz = float(final_epoch_match.group(2))
            else:
                final_acc = accuracy
                final_dtz = dtz_mae
            
            print(f"\nResults:")
            print(f"  Best validation accuracy: {accuracy:.2f}%")
            print(f"  Best validation DTZ MAE: {dtz_mae:.2f}")
            print(f"  Final epoch accuracy: {final_acc:.2f}%")
            print(f"  Final epoch DTZ MAE: {final_dtz:.2f}")
            
            return {
                'success': True,
                'endgame': endgame,
                'best_accuracy': accuracy,
                'best_dtz_mae': dtz_mae,
                'final_accuracy': final_acc,
                'final_dtz_mae': final_dtz,
                'dataset': dataset_path,
                'epochs': epochs
            }
        else:
            print(f"\nTraining failed!")
            return {
                'success': False,
                'endgame': endgame,
                'error': result.stderr[:500]
            }
            
    except subprocess.TimeoutExpired:
        print(f"\nTraining timed out")
        return {
            'success': False,
            'endgame': endgame,
            'error': 'Timeout'
        }

def main():
    print("="*60)
    print("CANONICAL FORMS TRAINING COMPARISON")
    print("="*60)
    
    # Original results (from encoding v2)
    original_results = {
        'KQvK': {'accuracy': 99.94, 'dtz_mae': 0.64},
        'KRvK': {'accuracy': 100.00, 'dtz_mae': 1.00},
        'KPvK': {'accuracy': 99.88, 'dtz_mae': 0.06}
    }
    
    # Train canonical models
    canonical_results = {}
    
    # Train with 50 epochs for reasonable comparison
    epochs = 50
    
    for endgame in ['KQvK', 'KRvK', 'KPvK']:
        dataset_path = f"data/{endgame}_canonical.npz"
        result = run_training(dataset_path, endgame, epochs=epochs)
        canonical_results[endgame] = result
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON: ORIGINAL vs CANONICAL")
    print(f"{'='*60}")
    
    comparisons = []
    
    for endgame in ['KQvK', 'KRvK', 'KPvK']:
        print(f"\n{endgame}:")
        
        orig = original_results[endgame]
        canon = canonical_results[endgame]
        
        if canon['success']:
            # Use final accuracy for comparison
            accuracy_diff = canon['final_accuracy'] - orig['accuracy']
            dtz_diff = canon['final_dtz_mae'] - orig['dtz_mae']
            
            print(f"  Original: {orig['accuracy']:.2f}% WDL, {orig['dtz_mae']:.2f} DTZ MAE")
            print(f"  Canonical: {canon['final_accuracy']:.2f}% WDL, {canon['final_dtz_mae']:.2f} DTZ MAE")
            print(f"  Δ Accuracy: {accuracy_diff:+.2f}%")
            print(f"  Δ DTZ MAE: {dtz_diff:+.2f}")
            
            comparisons.append({
                'endgame': endgame,
                'original_accuracy': orig['accuracy'],
                'canonical_accuracy': canon['final_accuracy'],
                'accuracy_diff': accuracy_diff,
                'original_dtz_mae': orig['dtz_mae'],
                'canonical_dtz_mae': canon['final_dtz_mae'],
                'dtz_diff': dtz_diff
            })
        else:
            print(f"  Training failed: {canon.get('error', 'Unknown error')}")
    
    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    
    results_file = f"results/canonical_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'original_results': original_results,
            'canonical_results': canonical_results,
            'comparisons': comparisons,
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Summary
    if comparisons:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        avg_accuracy_diff = sum(c['accuracy_diff'] for c in comparisons) / len(comparisons)
        avg_dtz_diff = sum(c['dtz_diff'] for c in comparisons) / len(comparisons)
        
        print(f"Average Δ Accuracy: {avg_accuracy_diff:+.2f}%")
        print(f"Average Δ DTZ MAE: {avg_dtz_diff:+.2f}")
        
        # Evaluation criteria
        if avg_accuracy_diff >= -1.0 and abs(avg_dtz_diff) <= 1.0:
            print(f"\n✓ Canonical forms perform SIMILARLY to original datasets")
            print(f"  (Accuracy difference: {avg_accuracy_diff:+.2f}%, DTZ difference: {avg_dtz_diff:+.2f})")
        elif avg_accuracy_diff > 0:
            print(f"\n✓ Canonical forms perform BETTER than original datasets!")
            print(f"  (Accuracy improvement: {avg_accuracy_diff:+.2f}%)")
        else:
            print(f"\n⚠ Canonical forms perform WORSE than original datasets")
            print(f"  (Accuracy reduction: {avg_accuracy_diff:+.2f}%)")

if __name__ == "__main__":
    main()