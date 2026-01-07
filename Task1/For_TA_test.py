import argparse
import torch
from dataloader import load_images_and_labels
from model import CNN
from utils import compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test images directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load saved model parameters
    state = torch.load('saved_model.pt', map_location=device)
    model = CNN(device)
    model.conv1_weight = state['conv1_weight'].to(device)
    model.conv1_bias = state['conv1_bias'].to(device)
    model.conv2_weight = state['conv2_weight'].to(device)
    model.conv2_bias = state['conv2_bias'].to(device)
    model.fc_weight = state['fc_weight'].to(device)
    model.fc_bias = state['fc_bias'].to(device)

    # Load test data
    test_data = load_images_and_labels(args.test_data_path, device)

    # Evaluate on test set
    y_true = []
    y_pred = []
    with torch.no_grad():
        for img, label in test_data:
            output = model.forward(img)
            pred_label = 1 if output.item() >= 0.5 else 0
            y_true.append(label)
            y_pred.append(pred_label)
    _, _, f1 = compute_metrics(y_true, y_pred)

    # Print student ID and F1 (format: ID:F1)
    student_id = 'PB23000000'  # replace with your actual student ID
    print(f'{student_id}:{f1:.2f}')

if __name__ == '__main__':
    main()
