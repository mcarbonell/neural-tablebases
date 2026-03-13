"""Plot training progress from log files"""
import re
import matplotlib.pyplot as plt

def parse_log(log_file):
    """Parse training log and extract metrics"""
    epochs = []
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    lr = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Time:' in line:
                # Extract epoch number
                epoch_match = re.search(r'Epoch (\d+)/\d+', line)
                if epoch_match:
                    epochs.append(int(epoch_match.group(1)))
                
                # Extract metrics
                train_acc_match = re.search(r'Train Acc: ([\d.]+)', line)
                val_acc_match = re.search(r'Val Acc: ([\d.]+)', line)
                train_loss_match = re.search(r'Train Loss: ([\d.]+)', line)
                val_loss_match = re.search(r'Val Loss: ([\d.]+)', line)
                lr_match = re.search(r'LR: ([\d.]+)', line)
                
                if train_acc_match:
                    train_acc.append(float(train_acc_match.group(1)))
                if val_acc_match:
                    val_acc.append(float(val_acc_match.group(1)))
                if train_loss_match:
                    train_loss.append(float(train_loss_match.group(1)))
                if val_loss_match:
                    val_loss.append(float(val_loss_match.group(1)))
                if lr_match:
                    lr.append(float(lr_match.group(1)))
    
    return epochs, train_acc, val_acc, train_loss, val_loss, lr

# Parse the latest log
log_file = 'logs/train_mlp_20260312_214059.log'
epochs, train_acc, val_acc, train_loss, val_loss, lr = parse_log(log_file)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MLP Training Progress (529K params, 3 classes)', fontsize=16)

# Plot 1: Accuracy
axes[0, 0].plot(epochs, train_acc, label='Train Acc', linewidth=2)
axes[0, 0].plot(epochs, val_acc, label='Val Acc', linewidth=2, alpha=0.7)
axes[0, 0].axhline(y=0.6, color='r', linestyle='--', alpha=0.3, label='60% baseline')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy over Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[0, 1].plot(epochs, train_loss, label='Train Loss', linewidth=2)
axes[0, 1].plot(epochs, val_loss, label='Val Loss', linewidth=2, alpha=0.7)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Loss over Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Learning Rate
axes[1, 0].plot(epochs, lr, linewidth=2, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Val Accuracy Distribution
axes[1, 1].hist(val_acc, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=sum(val_acc)/len(val_acc), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {sum(val_acc)/len(val_acc):.3f}')
axes[1, 1].set_xlabel('Validation Accuracy')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Val Accuracy Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
print("Plot saved to training_progress.png")

# Print summary statistics
print(f"\n=== Training Summary ===")
print(f"Total epochs: {len(epochs)}")
print(f"Best val accuracy: {max(val_acc):.4f} at epoch {epochs[val_acc.index(max(val_acc))]}")
print(f"Best val loss: {min(val_loss):.4f} at epoch {epochs[val_loss.index(min(val_loss))]}")
print(f"Final train acc: {train_acc[-1]:.4f}")
print(f"Final val acc: {val_acc[-1]:.4f}")
print(f"Val acc std dev: {(sum((x - sum(val_acc)/len(val_acc))**2 for x in val_acc) / len(val_acc))**0.5:.4f}")
print(f"Improvement from epoch 1: {val_acc[-1] - val_acc[0]:.4f} ({100*(val_acc[-1] - val_acc[0])/val_acc[0]:.1f}%)")
