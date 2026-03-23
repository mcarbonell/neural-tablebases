import matplotlib.pyplot as plt
import pandas as pd
import os

# Reconstructed data from terminal output
# (Epoch, Batch, Loss, Acc%)
training_points = [
    (1, 0, 14.9991, 37.322998),
    (1, 100, 2.8064, 85.574311),
    (1, 200, 2.2380, 87.792574),
    (1, 300, 1.9697, 88.944343),
    (1, 400, 1.9337, 89.743110),
    (1, 2100, 1.2853, 93.257809),
    (5, 500, 0.7431, 97.008924),
    (5, 2100, 0.7209, 97.054515),
    (6, 0, 0.7040, 97.174072),
    (6, 2100, 0.7285, 97.162580),
    (7, 0, 0.6560, 97.174072),
    (7, 2100, 0.6618, 97.238405),
    (8, 0, 0.6458, 97.259521),
    (8, 2100, 0.6553, 97.303914),
    (9, 0, 0.6314, 97.406006),
    (9, 1200, 0.6150, 97.340463)
]

# Validation points (at end of epoch)
val_points = [
    (1, 96.06),
    (5, 97.59),
    (6, 97.68),
    (7, 97.73),
    (8, 97.78)
]

def plot_progress():
    df = pd.DataFrame(training_points, columns=['Epoch', 'Batch', 'Loss', 'Acc'])
    df['Global_Step'] = (df['Epoch'] - 1) * 2198 + df['Batch']
    
    val_df = pd.DataFrame(val_points, columns=['Epoch', 'ValAcc'])
    val_df['Global_Step'] = val_df['Epoch'] * 2198
    
    plt.figure(figsize=(10, 6))
    
    # Plot Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(df['Global_Step'], df['Acc'], label='Training Acc %', marker='.')
    plt.scatter(val_df['Global_Step'], val_df['ValAcc'], color='red', label='Val Acc %', zorder=5)
    plt.title('Universal V6 Model - Training Progress')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Loss
    plt.subplot(2, 1, 2)
    plt.plot(df['Global_Step'], df['Loss'], label='Training Loss', color='orange')
    plt.ylabel('Loss')
    plt.xlabel('Global Step (Batches)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'docs/results/training_v6_progress_reconstructed.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    plot_progress()
