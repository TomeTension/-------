import os
from PIL import Image
import torch


def load_images_and_labels(data_path, device):
    """
    data_path/
      ├── img/
      └── txt/
    """
    img_dir = os.path.join(data_path, "img")
    txt_dir = os.path.join(data_path, "txt")

    if not os.path.isdir(img_dir):
        raise RuntimeError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(txt_dir):
        raise RuntimeError(f"Label directory not found: {txt_dir}")

    data = []
    img_ext = ('.png', '.jpg', '.jpeg', '.bmp')

    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(img_ext):
            continue

        img_path = os.path.join(img_dir, fname)
        base, _ = os.path.splitext(fname)
        txt_path = os.path.join(txt_dir, base + '.txt')

        # YOLO 语义：txt 存在但为空 = 无缺陷（合法样本）
        if not os.path.exists(txt_path):
            continue  # 或 raise，看你数据是否保证一一对应

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        label = 1 if len(lines) > 0 else 0

        img = Image.open(img_path).convert('L')
        img = img.resize((224, 224))

        pixels = torch.tensor(list(img.getdata()), dtype=torch.float32) / 255.0
        img_tensor = pixels.view(224, 224).unsqueeze(0).to(device)

        data.append((img_tensor, label))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data
