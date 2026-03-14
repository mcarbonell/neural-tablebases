"""
Simple script to train models on canonical datasets.
"""
import subprocess
import sys
import json
from datetime import datetime

def train_single_model(dataset_path, endgame_name):
    """Train a single model and return results."""
    print(f"\n{'='*60}")
    print(f"Training {endgame_name} canonical")
    print(f"{'='*60}")
    
    # Command with reduced epochs for faster testing
    cmd = [
        sys.executable, "src/train.py",
        "--data_path", dataset_path,
        "--epochs", "100",  # Reduced for faster testing
        "--batch_size", "1024",
        "--lr", "0.001",
        "--model", "mlp",
        "--hard_mining"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print("\nTraining completed successfully!")
            
            # Parse results
            accuracy = None
            dtz_mae = None
            
            for line in result.stdout.split('\n'):
                if "Final WDL Accuracy" in line:
                    try:
                        accuracy = float(line.split(':')[1].strip().replace('%', ''))
                        print(f"WDL Accuracy: {accuracy}%")
                    except:
                        pass
                elif "Final DTZ MAE" in line:
                    try:
                        dtz_mae = float(line.split(':')[1].strip())
                        print(f"DTZ MAE: {dtz_mae}")
                    except:
                        pass
            
            return {
                'success': True,
                'endgame': endgame_name,
                'accuracy': accuracy,
                'dtz_mae': dtz_mae,
                'dataset_size': dataset_path
            }
        else:
            print(f"\nTraining failed!")
            print(f"Error: {result.stderr[:500]}")
            return {
                'success': False,
                'endgame': endgame_name,
                'error': result.stderr[:500]
            }
            
    except subprocess.TimeoutExpired:
        print(f"\nTraining timed out after 30 minutes")
        return {
            'success': False,
            'endgame': endgame_name,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"\nException: {e}")
        return {
            'success': False,
            'endgame': endgame_name,
            'error': str(e)
        }

def main():
    print("="*60)
    print("TRAINING CANONICAL DATASETS")
    print("="*60)
    
    results = {}
    
    # Train KQvK canonical
    results['KQvK'] = train_single_model("data/KQvK_canonical.npz", "KQvK")
    
    # Train KRvK canonical  
    results['KRvK'] = train_single_model("data/KRvK_canonical.npz", "KRvK")
    
    # Train KPvK canonical
    results['KPvK'] = train_single_model("data/KPvK_canonical.npz", "KPvK")
    
    # Save results
    results_file = "results/canonical_training_simple.json"
    import os
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    # Compare with original results
    original_results = {
        'KQvK': {'accuracy': 99.94, 'dtz_mae': 0.64},
        'KRvK': {'accuracy': 100.00, 'dtz_mae': 1.00},
        'KPvK': {'accuracy': 99.88, 'dtz_mae': 0.06}
    }
    
    for endgame in ['KQvK', 'KRvK', 'KPvK']:
        print(f"\n{endgame}:")
        if results[endgame]['success']:
            orig = original_results[endgame]
            canon = results[endgame]
            
            accuracy_diff = canon['accuracy'] - orig['accuracy']
            dtz_diff = canon['dtz_mae'] - orig['dtz_mae']
            
            print(f"  Original: {orig['accuracy']:.2f}% WDL, {orig['dtz_mae']:.2f} DTZ MAE")
            print(f"  Canonical: {canon['accuracy']:.2f}% WDL, {canon['dtz_mae']:.2f} DTZ MAE")
            print(f"  Δ Accuracy: {accuracy_diff:+.2f}%")
            print(f"  Δ DTZ MAE: {dtz_diff:+.2f}")
        else:
            print(f"  Training failed: {results[endgame].get('error', 'Unknown error')}")
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()