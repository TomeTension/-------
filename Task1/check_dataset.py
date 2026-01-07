import os

img_dir = r"E:\courses\机器学习\project\dataset\small_train\img"
txt_dir = r"E:\courses\机器学习\project\dataset\small_train\txt"

img_files = sorted(os.listdir(img_dir))
txt_files = sorted(os.listdir(txt_dir))

img_basenames = set(os.path.splitext(f)[0] for f in img_files)
txt_basenames = set(os.path.splitext(f)[0] for f in txt_files)

common = img_basenames & txt_basenames

print("Total images:", len(img_basenames))
print("Total txts:", len(txt_basenames))
print("Matched pairs:", len(common))

print("\nExamples of unmatched images:")
for name in list(img_basenames - txt_basenames)[:10]:
    print("  ", name)

print("\nExamples of unmatched txts:")
for name in list(txt_basenames - img_basenames)[:10]:
    print("  ", name)
