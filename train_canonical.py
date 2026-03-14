"""
Train models on canonical datasets and compare with original results.
"""
import subprocess
import time
import os
import sys
import json
from datetime import datetime

def train_model(dataset_path, model_name, epochs=100, batch_size=1024, learning_rate=0.001):
    """Train a model on a dataset."""
    print(f"\nTraining {model_name} on {dataset_path}...")
    
    # Create output directory
    output_dir = f"models/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Log file
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Command (using correct arguments from train.py)
    cmd = [
        sys.executable, "src/train.py",
        "--data_path", dataset_path,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate),
        "--model", "mlp",
        "--wdl_classes", "3",
        "--hard_mining"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Run training
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  text=True, timeout=3600)  # 1 hour timeout
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Training completed in {elapsed:.1f}s")
            
            # Parse results from output
            lines = result.stdout.split('\n')
            final_accuracy = None
            final_dtz_mae = None
            
            for line in lines:
                if "Final WDL Accuracy" in line:
                    final_accuracy = float(line.split(':')[1].strip().replace('%', ''))
                elif "Final DTZ MAE" in line:
                    final_dtz_mae = float(line.split(':')[1].strip())
            
            return {
                'success': True,
                'model_name': model_name,
                'dataset': dataset_path,
                'accuracy': final_accuracy,
                'dtz_mae': final_dtz_mae,
                'time': elapsed,
                'log_file': log_file
            }
        else:
            print(f"✗ Training failed in {elapsed:.1f}s")
            print(f"Error output (last 500 chars):\n{result.stdout[-500:]}")
            return {
                'success': False,
                'model_name': model_name,
                'error': result.stdout[-500:]
            }
            
    except subprocess.TimeoutExpired:
        print(f"✗ Training timed out after 3600 seconds")
        return {
            'success': False,
            'model_name': model_name,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"✗ Exception: {e}")
        return {
            'success': False,
            'model_name': model_name,
            'error': str(e)
        }

def compare_results(original_results, canonical_results):
    """Compare original vs canonical results."""
    print("\n" + "="*60)
    print("COMPARISON: ORIGINAL vs CANONICAL")
    print("="*60)
    
    comparisons = []
    
    for endgame in ['KQvK', 'KRvK', 'KPvK']:
        orig_key = f"{endgame}_original"
        canon_key = f"{endgame}_canonical"
        
        if orig_key in original_results and canon_key in canonical_results:
            orig = original_results[orig_key]
            canon = canonical_results[canon_key]
            
            if orig['success'] and canon['success']:
                accuracy_diff = canon['accuracy'] - orig['accuracy']
                dtz_diff = canon['dtz_mae'] - orig['dtz_mae']
                
                print(f"\n{endgame}:")
                print(f"  Original: {orig['accuracy']:.2f}% WDL, {orig['dtz_mae']:.2f} DTZ MAE")
                print(f"  Canonical: {canon['accuracy']:.2f}% WDL, {canon['dtz_mae']:.2f} DTZ MAE")
                print(f"  Δ Accuracy: {accuracy_diff:+.2f}%")
                print(f"  Δ DTZ MAE: {dtz_diff:+.2f}")
                
                comparisons.append({
                    'endgame': endgame,
                    'original_accuracy': orig['accuracy'],
                    'canonical_accuracy': canon['accuracy'],
                    'accuracy_diff': accuracy_diff,
                    'original_dtz_mae': orig['dtz_mae'],
                    'canonical_dtz_mae': canon['dtz_mae'],
                    'dtz_diff': dtz_diff
                })
    
    return comparisons

def main():
    print("="*60)
    print("CANONICAL FORMS TRAINING COMPARISON")
    print("="*60)
    
    # Known original results (from previous training)
    original_results = {
        'KQvK_original': {
            'success': True,
            'accuracy': 99.94,  # From encoding v2 results
            'dtz_mae': 0.64
        },
        'KRvK_original': {
            'success': True,
            'accuracy': 100.00,
            'dtz_mae': 1.00
        },
        'KPvK_original': {
            'success': True,
            'accuracy': 99.88,
            'dtz_mae': 0.06
        }
    }
    
    # Train on canonical datasets
    canonical_results = {}
    
    # Training parameters (reduced for quick testing)
    epochs = 50  # Reduced from 100 for faster testing
    batch_size = 1024
    learning_rate = 0.001
    
    # Train KQvK canonical
    result = train_model(
        dataset_path="data/KQvK_canonical.npz",
        model_name="kqvk_canonical",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    canonical_results['KQvK_canonical'] = result
    
    # Train KRvK canonical
    result = train_model(
        dataset_path="data/KRvK_canonical.npz",
        model_name="krvk_canonical",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    canonical_results['KRvK_canonical'] = result
    
    # Train KPvK canonical
    result = train_model(
        dataset_path="data/KPvK_canonical.npz",
        model_name="kpvk_canonical",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    canonical_results['KPvK_canonical'] = result
    
    # Compare results
    comparisons = compare_results(original_results, canonical_results)
    
    # Save results
    results_file = "results/canonical_training_results.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'original_results': original_results,
            'canonical_results': canonical_results,
            'comparisons': comparisons,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if comparisons:
        avg_accuracy_diff = sum(c['accuracy_diff'] for c in comparisons) / len(comparisons)
        avg_dtz_diff = sum(c['dtz_diff'] for c in comparisons) / len(comparisons)
        
        print(f"Average Δ Accuracy: {avg_accuracy_diff:+.2f}%")
        print(f"Average Δ DTZ MAE: {avg_dtz_diff:+.2f}")
        
        if avg_accuracy_diff >= -0.5 and avg_dtz_diff <= 0.5:  # Small differences acceptable
            print("\n✓ Canonical forms perform similarly to original datasets")
            print("  (Differences within acceptable range)")
        else:
            print("\n⚠ Canonical forms show significant differences")
            print("  (May need further investigation)")
    else:
        print("No comparisons available (some trainings may have failed)")
    
    return canonical_results

if __name__ == "__main__":
    results = main()