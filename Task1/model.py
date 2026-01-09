import torch
import torch.nn.functional as F

class CNN:
    def __init__(self, device):
        self.device = device
        
        # 为了适配你的 main.py，必须使用这些变量名
        # 虽然我们有6层卷积，但 main.py 可能只存取部分。
        # 我们将主要权重命名为 conv1, conv2, fc，其余作为辅助。

        def init_w(out_c, in_c, k):
            return torch.randn(out_c, in_c, k, k, device=device) * (2. / (in_c * k * k))**0.5

        # --- Block 1 ---
        self.conv1_weight = init_w(16, 1, 3) # 改回原名
        self.conv1_bias = torch.zeros(16, device=device)
        self.c2_w = init_w(16, 16, 3)
        self.c2_b = torch.zeros(16, device=device)

        # --- Block 2 ---
        self.conv2_weight = init_w(32, 16, 3) # 改回原名
        self.conv2_bias = torch.zeros(32, device=device)
        self.c4_w = init_w(32, 32, 3)
        self.c4_b = torch.zeros(32, device=device)

        # --- Block 3 ---
        self.c5_w = init_w(64, 32, 3)
        self.c5_b = torch.zeros(64, device=device)
        self.c6_w = init_w(64, 64, 3)
        self.c6_b = torch.zeros(64, device=device)

        # --- FC ---
        self.flat_dim = 64 * 28 * 28
        self.fc_weight = torch.randn(self.flat_dim, 1, device=device) * 0.01 # 改回原名
        self.fc_bias = torch.zeros(1, device=device)

    def forward(self, x):
        self.x = x
        # Block 1
        self.z1 = F.conv2d(x, self.conv1_weight, self.conv1_bias, padding=1)
        self.a1 = F.relu(self.z1)
        self.z2 = F.conv2d(self.a1, self.c2_w, self.c2_b, padding=1)
        self.a2 = F.relu(self.z2)
        self.p1, self.idx1 = F.max_pool2d(self.a2, 2, return_indices=True)

        # Block 2
        self.z3 = F.conv2d(self.p1, self.conv2_weight, self.conv2_bias, padding=1)
        self.a3 = F.relu(self.z3)
        self.z4 = F.conv2d(self.a3, self.c4_w, self.c4_b, padding=1)
        self.a4 = F.relu(self.z4)
        self.p2, self.idx2 = F.max_pool2d(self.a4, 2, return_indices=True)

        # Block 3
        self.z5 = F.conv2d(self.p2, self.c5_w, self.c5_b, padding=1)
        self.a5 = F.relu(self.z5)
        self.z6 = F.conv2d(self.a5, self.c6_w, self.c6_b, padding=1)
        self.a6 = F.relu(self.z6)
        self.p3, self.idx3 = F.max_pool2d(self.a6, 2, return_indices=True)

        self.flat = self.p3.view(x.shape[0], -1)
        self.z_fc = self.flat.matmul(self.fc_weight) + self.fc_bias
        self.out = torch.sigmoid(self.z_fc).squeeze(1)
        return self.out

    def backward(self, y, lr):
        B = y.shape[0]
        y = y.view(-1, 1)
        out = self.out.view(-1, 1)

        # 稍微调低 pos_weight (从4.0降到2.5)，平衡 Precision 和 Recall
        pos_weight = 2.5 
        weights = torch.where(y == 1, torch.tensor(pos_weight, device=self.device), torch.tensor(1.0, device=self.device))
        dz_fc = (out - y) * weights

        dw_fc = self.flat.t().matmul(dz_fc) / B
        db_fc = dz_fc.mean(dim=0)
        d_p3 = dz_fc.matmul(self.fc_weight.t()).view(self.p3.shape)

        # Block 3 Back
        d_a6 = F.max_unpool2d(d_p3, self.idx3, 2, output_size=self.a6.shape)
        d_z6 = d_a6 * (self.z6 > 0).float()
        dw6 = F.conv2d(self.a5.transpose(0,1), d_z6.transpose(0,1), padding=1).transpose(0,1) / B
        db6 = d_z6.sum(dim=(0,2,3)) / B
        d_a5 = F.conv_transpose2d(d_z6, self.c6_w, padding=1)

        d_z5 = d_a5 * (self.z5 > 0).float()
        dw5 = F.conv2d(self.p2.transpose(0,1), d_z5.transpose(0,1), padding=1).transpose(0,1) / B
        db5 = d_z5.sum(dim=(0,2,3)) / B
        d_p2 = F.conv_transpose2d(d_z5, self.c5_w, padding=1)

        # Block 2 Back
        d_a4 = F.max_unpool2d(d_p2, self.idx2, 2, output_size=self.a4.shape)
        d_z4 = d_a4 * (self.z4 > 0).float()
        dw4 = F.conv2d(self.a3.transpose(0,1), d_z4.transpose(0,1), padding=1).transpose(0,1) / B
        db4 = d_z4.sum(dim=(0,2,3)) / B
        d_a3 = F.conv_transpose2d(d_z4, self.c4_w, padding=1)

        d_z3 = d_a3 * (self.z3 > 0).float()
        dw3 = F.conv2d(self.p1.transpose(0,1), d_z3.transpose(0,1), padding=1).transpose(0,1) / B
        db3 = d_z3.sum(dim=(0,2,3)) / B
        d_p1 = F.conv_transpose2d(d_z3, self.conv2_weight, padding=1)

        # Block 1 Back
        d_a2 = F.max_unpool2d(d_p1, self.idx1, 2, output_size=self.a2.shape)
        d_z2 = d_a2 * (self.z2 > 0).float()
        dw2 = F.conv2d(self.a1.transpose(0,1), d_z2.transpose(0,1), padding=1).transpose(0,1) / B
        db2 = d_z2.sum(dim=(0,2,3)) / B
        d_a1 = F.conv_transpose2d(d_z2, self.c2_w, padding=1)

        d_z1 = d_a1 * (self.z1 > 0).float()
        dw1 = F.conv2d(self.x.transpose(0,1), d_z1.transpose(0,1), padding=1).transpose(0,1) / B
        db1 = d_z1.sum(dim=(0,2,3)) / B

        # 更新权重
        self.fc_weight -= lr * dw_fc; self.fc_bias -= lr * db_fc
        self.c6_w -= lr * dw6; self.c6_b -= lr * db6
        self.c5_w -= lr * dw5; self.c5_b -= lr * db5
        self.c4_w -= lr * dw4; self.c4_b -= lr * db4
        self.conv2_weight -= lr * dw3; self.conv2_bias -= lr * db3
        self.c2_w -= lr * dw2; self.c2_b -= lr * db2
        self.conv1_weight -= lr * dw1; self.conv1_bias -= lr * db1