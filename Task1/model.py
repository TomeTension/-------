import torch
import torch.nn.functional as F


class CNN:
    """
    Shallow CNN for binary classification (Task 1)
    Architecture:
        Conv(1→4) → ReLU →
        Conv(4→8) → ReLU →
        Flatten → FC → Sigmoid

    All gradients are manually derived.
    No autograd is used.
    """

    def __init__(self, device):
        self.device = device

        # ---------- Conv layer 1 ----------
        self.conv1_weight = torch.randn(4, 1, 3, 3, device=device) * 0.01
        self.conv1_bias = torch.zeros(4, device=device)

        # ---------- Conv layer 2 ----------
        self.conv2_weight = torch.randn(8, 4, 3, 3, device=device) * 0.01
        self.conv2_bias = torch.zeros(8, device=device)

        # Input: 224×224
        # conv1 → 222×222
        # conv2 → 220×220
        self.fc_in_dim = 8 * 220 * 220

        # ---------- Fully connected ----------
        self.fc_weight = torch.randn(self.fc_in_dim, device=device) * 0.01
        self.fc_bias = torch.zeros(1, device=device)

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, x):
        """
        x: [1, 224, 224] grayscale image
        return: scalar sigmoid output
        """
        x = x.unsqueeze(0)  # [1,1,224,224]

        # Conv1 + ReLU
        self.conv1 = F.conv2d(x, self.conv1_weight, self.conv1_bias)
        self.relu1 = torch.clamp(self.conv1, min=0)

        # Conv2 + ReLU
        self.conv2 = F.conv2d(self.relu1, self.conv2_weight, self.conv2_bias)
        self.relu2 = torch.clamp(self.conv2, min=0)

        # Flatten
        self.flat = self.relu2.view(-1)

        # FC + Sigmoid
        self.fc_out = self.flat.matmul(self.fc_weight) + self.fc_bias
        self.output = torch.sigmoid(self.fc_out)

        self.x = x
        return self.output

    # ============================================================
    # Backward (manual)
    # ============================================================
    def backward(self, label, lr):
        """
        label: 0 or 1
        lr: learning rate
        """
        y = torch.tensor(label, device=self.device, dtype=torch.float32)
        out = self.output.squeeze()

        # --------------------------------------------------------
        # BCE + Sigmoid gradient
        # dL/dz = (σ(z) - y)
        # --------------------------------------------------------
        grad_fc_out = out - y  # scalar

        # ---------- FC gradients ----------
        grad_fc_bias = grad_fc_out
        grad_fc_weight = grad_fc_out * self.flat

        grad_flat = grad_fc_out * self.fc_weight
        grad_relu2 = grad_flat.view(1, 8, 220, 220)

        # --------------------------------------------------------
        # ReLU2 backward
        # --------------------------------------------------------
        grad_conv2 = grad_relu2.clone()
        grad_conv2[self.conv2 <= 0] = 0

        # ---------- Conv2 gradients ----------
        grad_conv2_weight = torch.zeros_like(self.conv2_weight)
        grad_conv2_bias = torch.zeros_like(self.conv2_bias)

        for oc in range(8):
            grad_conv2_bias[oc] = grad_conv2[0, oc].sum()
            for ic in range(4):
                grad_conv2_weight[oc, ic] = F.conv2d(
                    self.relu1[:, ic:ic+1],
                    grad_conv2[:, oc:oc+1]
                ).squeeze()

        # ---------- Propagate to conv1 ----------
        grad_relu1 = F.conv_transpose2d(grad_conv2, self.conv2_weight)

        # --------------------------------------------------------
        # ReLU1 backward
        # --------------------------------------------------------
        grad_conv1 = grad_relu1.clone()
        grad_conv1[self.conv1 <= 0] = 0

        # ---------- Conv1 gradients ----------
        grad_conv1_weight = torch.zeros_like(self.conv1_weight)
        grad_conv1_bias = torch.zeros_like(self.conv1_bias)

        for oc in range(4):
            grad_conv1_bias[oc] = grad_conv1[0, oc].sum()
            grad_conv1_weight[oc, 0] = F.conv2d(
                self.x,
                grad_conv1[:, oc:oc+1]
            ).squeeze()

        # --------------------------------------------------------
        # SGD update
        # --------------------------------------------------------
        self.fc_weight.data -= lr * grad_fc_weight
        self.fc_bias.data -= lr * grad_fc_bias

        self.conv2_weight.data -= lr * grad_conv2_weight
        self.conv2_bias.data -= lr * grad_conv2_bias

        self.conv1_weight.data -= lr * grad_conv1_weight
        self.conv1_bias.data -= lr * grad_conv1_bias
