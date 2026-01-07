import argparse
import torch
from dataloader import load_images_and_labels
from model import CNN
from utils import compute_metrics

def main():
    # Parse optional arguments for data paths
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, default='train', help='Path to training images directory')
    parser.add_argument('--val_data_path', type=str, default='val', help='Path to validation images directory')
    args = parser.parse_args()

    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training and validation data
    train_data = load_images_and_labels(args.train_data_path, device)
    val_data = load_images_and_labels(args.val_data_path, device)

    # Initialize model
    model = CNN(device)

    # Training hyperparameters
    num_epochs = 5     # For demonstration; adjust as needed
    lr = 0.001         # Learning rate for SGD

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        # Training loop
        for img, label in train_data:
            # Forward pass
            output = model.forward(img)  # sigmoid output (tensor of shape [1])
            prob = output.clamp(1e-6, 1 - 1e-6)  # clamp for numerical stability
            # Compute BCE loss
            y_tensor = torch.tensor(label, device=device, dtype=torch.float32)
            loss = -(y_tensor * torch.log(prob) + (1 - y_tensor) * torch.log(1 - prob))
            total_loss += loss.item()
            # Prediction (threshold 0.5)
            pred_label = 1 if output.item() >= 0.5 else 0
            if pred_label == label:
                correct += 1
            # Backward and update
            model.backward(label, lr)

        # Compute training metrics
        avg_loss = total_loss / len(train_data)
        accuracy = correct / len(train_data)

        # Validation metrics
        y_true_val = []
        y_pred_val = []
        with torch.no_grad():
            for img, label in val_data:
                output = model.forward(img)
                pred_label = 1 if output.item() >= 0.5 else 0
                y_true_val.append(label)
                y_pred_val.append(pred_label)
        precision, recall, f1 = compute_metrics(y_true_val, y_pred_val)

        # Print epoch results
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}, '
              f'Val Precision={precision:.4f}, Val Recall={recall:.4f}, Val F1={f1:.4f}')

    # Save trained model parameters
    torch.save({
        'conv1_weight': model.conv1_weight.cpu(),
        'conv1_bias': model.conv1_bias.cpu(),
        'conv2_weight': model.conv2_weight.cpu(),
        'conv2_bias': model.conv2_bias.cpu(),
        'fc_weight': model.fc_weight.cpu(),
        'fc_bias': model.fc_bias.cpu()
    }, 'saved_model.pt')

if __name__ == '__main__':
    main()
