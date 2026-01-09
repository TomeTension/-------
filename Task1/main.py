import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import DefectDataset
from model import CNN
from utils import compute_metrics


def main():
    # -------------------------------
    # Parse arguments
    # -------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    args = parser.parse_args()

    # -------------------------------
    # Set device
    # -------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # -------------------------------
    # Load dataset & dataloader
    # -------------------------------
    train_dataset = DefectDataset(args.train_data_path, augment=True)
    val_dataset = DefectDataset(args.val_data_path, augment=False)

    batch_size = 128
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------------
    # Initialize model
    # -------------------------------
    model = CNN(device)

    # -------------------------------
    # Hyperparameters
    # -------------------------------
    num_epochs = 20
    lr = 1e-3

    # -------------------------------
    # Metric storage (for plotting)
    # -------------------------------
    train_losses, train_accs = [], []
    val_precisions, val_recalls, val_f1s = [], [], []

    # -------------------------------
    # Open score.txt
    # -------------------------------
    with open('score.txt', 'w') as f:
        f.write("Model Hyperparameters:\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n\n")
        f.write("Training process:\n")

        # -------------------------------
        # Training loop
        # -------------------------------
        for epoch in range(num_epochs):
            model_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                unit="batch"
            )

            for xb, yb in pbar:
                xb = xb.to(device)
                yb = yb.to(device)

                # Forward
                out = model.forward(xb)

                # BCE loss (mean)
                loss = -(yb * torch.log(out + 1e-6) +
                         (1 - yb) * torch.log(1 - out + 1e-6)).mean()

                # Backward (manual)
                model.backward(yb, lr)

                # Statistics
                batch_size_now = yb.size(0)
                model_loss += loss.item() * batch_size_now
                preds = (out >= 0.5).float()
                correct += (preds == yb).sum().item()
                total += batch_size_now

                pbar.set_postfix(loss=loss.item())

            avg_loss = model_loss / total
            accuracy = correct / total
            train_losses.append(avg_loss)
            train_accs.append(accuracy)

            # -------------------------------
            # Validation
            # -------------------------------
            y_true, y_pred = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    out = model.forward(xb)
                    preds = (out >= 0.5).int().cpu().tolist()
                    y_pred.extend(preds)
                    y_true.extend(yb.int().tolist())

            precision, recall, f1 = compute_metrics(y_true, y_pred)
            val_precisions.append(precision)
            val_recalls.append(recall)
            val_f1s.append(f1)

            log_line = (
                f"Epoch {epoch+1}: "
                f"Loss={avg_loss:.4f}, Acc={accuracy:.4f}, "
                f"Val Precision={precision:.4f}, "
                f"Val Recall={recall:.4f}, Val F1={f1:.4f}\n"
            )
            print(log_line, end='')
            f.write(log_line)

    # -------------------------------
    # Save model
    # -------------------------------
    torch.save({
        'conv1_weight': model.conv1_weight.cpu(),
        'conv1_bias': model.conv1_bias.cpu(),
        'conv2_weight': model.conv2_weight.cpu(),
        'conv2_bias': model.conv2_bias.cpu(),
        'fc_weight': model.fc_weight.cpu(),
        'fc_bias': model.fc_bias.cpu()
    }, 'saved_model.pt')

    # -------------------------------
    # Plot metrics
    # -------------------------------
    epochs = list(range(1, num_epochs + 1))
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='x')
    plt.plot(epochs, val_precisions, label='Val Precision', marker='s')
    plt.plot(epochs, val_recalls, label='Val Recall', marker='d')
    plt.plot(epochs, val_f1s, label='Val F1', marker='^')

    plt.title('Training & Validation Metrics per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics_pretty.png')
    plt.show()


if __name__ == '__main__':
    main()

