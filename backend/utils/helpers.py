"""
Utility helpers for TensorFlow 2.19 training.
Replaces the old PyTorch save_checkpoint / load_checkpoint functions.
"""

import os
import matplotlib.pyplot as plt


def plot_training_curves(history, save_path=None):
    """
    Plot accuracy and loss curves from a Keras History object.

    Args:
        history : returned by model.fit()
        save_path : optional path to save the figure as PNG
    """
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, loss,     label='Train Loss')
    ax1.plot(epochs, val_loss, label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs, acc,     label='Train Acc')
    ax2.plot(epochs, val_acc, label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")

    plt.show()


def print_training_summary(history):
    """Print best train/val accuracy and loss from history."""
    best_val_acc  = max(history.history['val_accuracy']) * 100
    best_val_loss = min(history.history['val_loss'])
    final_epoch   = len(history.history['accuracy'])

    print("\n" + "="*40)
    print("TRAINING SUMMARY")
    print("-"*40)
    print(f"Epochs run       : {final_epoch}")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Best val loss    : {best_val_loss:.4f}")
    print("="*40)