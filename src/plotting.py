import matplotlib
matplotlib.use('Agg')  # non-interactive backend, needed for headless Oscar batch jobs
import matplotlib.pyplot as plt

def plot_training_curves(history, save_path="training_curve.png"):
    """Plot loss and Dice curves over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # losses
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()

    # validation dice
    ax2.plot(epochs, history['val_dice'], label='Val Dice', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice')
    ax2.set_title('Validation Dice Score')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curve to {save_path}")