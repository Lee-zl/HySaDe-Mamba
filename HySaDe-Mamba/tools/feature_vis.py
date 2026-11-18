import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
sys.path.append("DMSSN")
from torch.utils.data import DataLoader
from assistant.dataloader import HL_SC
from models.DSST import ACEN  # 使用你的模型
from assistant import pytorch_ssim, pytorch_iou
from sklearn.decomposition import PCA


# ========== 参数 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# target_layer_name = 'decoder'  # 修改为你要可视化的层名（支持 model.named_modules() 查看）
model_weights = "/home/zhanglei/PycharmProjects/HyperMamba_v2/DMSSN/netdir/HyperMamba1280_newdata_0005_2_SEB/HyperMamba1280_newdata_0005_2_SEB_epoch_95.pth"  # 修改为你的模型路径
save_dir = "./feature_visualization6"
os.makedirs(save_dir, exist_ok=True)

# ========== 数据 ==========
test_lists_path = '/home/zhanglei/PycharmProjects/HyperMamba_v2/train_lists.txt'
sc_dir = '/home/zhanglei/PycharmProjects/DMSSN-main_7.27_/DMSSN/SC_out/'
hlsod_dir = '/home/zhanglei/Datasets/HSODBIT-V1/ground_truth/'

with open(test_lists_path, 'r') as f:
    test_list = [line.strip() for line in f.readlines()]

test_dataset = HL_SC(img_list=test_list, img_path=sc_dir, lab_path=hlsod_dir, trans=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ========== 模型 ==========
model = ACEN().to(device)
ckpt = torch.load(model_weights)
# 处理 state_dict（如果是完整checkpoint，提取 state_dict）
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# 去掉 module. 前缀
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_k = k[7:]  # 去掉 module.
    else:
        new_k = k
    new_state_dict[new_k] = v

# 加载模型
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

# missing_keys, unexpected_keys = model.load_state_dict(checkpoint['module'], strict=True)
if missing_keys:
    print("Missing keys:", missing_keys)
if unexpected_keys:
    print("Unexpected keys:", unexpected_keys)
model.eval()

# # ========== Hook 设置 ==========
# feature_maps = []

# def hook_fn(module, input, output):
#     for out in output:  # output 是一个 tuple
#         feature_maps.append(out.detach().cpu())


# # 注册 hook
# for name, module in model.named_modules():
#     if name == target_layer_name:
#         module.register_forward_hook(hook_fn)
#         print(f"Hook registered on: {name}")

# # ========== 可视化函数 ==========
def normalize(tensor):
    t_min, t_max = tensor.min(), tensor.max()
    return (tensor - t_min) / (t_max - t_min + 1e-5)

def hyperspectral_to_rgb(hsi_image):
    h, w, b = hsi_image.shape
    reshaped = hsi_image.reshape(-1, b)  # [H*W, B]

    pca = PCA(n_components=3)
    rgb_flat = pca.fit_transform(reshaped)

    rgb_img = rgb_flat.reshape(h, w, 3)
    rgb_img = cv2.normalize(rgb_img, None, 0, 255, cv2.NORM_MINMAX)
    return rgb_img.astype(np.uint8)

# def visualize_feature_on_image(feature, original_image, save_path):
#     fmap = torch.mean(feature, dim=1).squeeze()  # 平均通道，得到热图
#     fmap = normalize(fmap)
#     fmap_np = fmap.numpy()
#     heatmap = cv2.applyColorMap(np.uint8(255 * fmap_np), cv2.COLORMAP_JET)
    
#     # 原图处理
#     original_np = original_image.permute(1, 2, 0).cpu().numpy()
#     # band_r, band_g, band_b = 30, 20, 10  # 示例波段索引
#     # rgb_img = np.stack([
#     #     original_np[:, :, band_r],
#     #     original_np[:, :, band_g],
#     #     original_np[:, :, band_b]
#     # ], axis=-1)
    
#     rgb_img = hyperspectral_to_rgb(original_np)

#     # 归一化到 0~255 并转为 uint8
#     rgb_img = cv2.normalize(rgb_img, None, 0, 255, cv2.NORM_MINMAX)
#     rgb_img = rgb_img.astype(np.uint8)
    
#     # 重采样至相同尺寸
#     heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
#     if heatmap_resized.shape[-1] != 3:
#         heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_GRAY2BGR)

#     overlay = cv2.addWeighted(rgb_img, 0.4, heatmap_resized, 0.6, 0)
#     cv2.imwrite(save_path, overlay)

# # ========== 可视化过程 ==========
# for idx, batch in enumerate(test_loader):
#     feature_maps.clear()
#     input_tensor = batch['image'].to(device)
#     name = batch['id'][0]

#     with torch.no_grad():
#         _ = model(input_tensor)

#     if not feature_maps:
#         print("No feature map captured.")
#         break

#     fmap = feature_maps[0]
#     save_path = os.path.join(save_dir, f"{name}_heatmap.jpg")
#     visualize_feature_on_image(fmap, input_tensor[0], save_path)
#     print(f"Saved: {save_path}")

# print("Feature map visualization completed.")



def visualize_feature_on_image(feature, original_image, save_path):
    fmap = torch.mean(feature, dim=1).squeeze()  # 平均通道，得到热图 (H, W)
    fmap = normalize(fmap)
    fmap_np = fmap.cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * fmap_np), cv2.COLORMAP_JET)

    original_np = original_image.permute(1, 2, 0).cpu().numpy()
    rgb_img = hyperspectral_to_rgb(original_np)

    rgb_img = cv2.normalize(rgb_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
    overlay = cv2.addWeighted(rgb_img, 0.4, heatmap_resized, 0.6, 0)
    cv2.imwrite(save_path, overlay)



# ========== 可视化过程 ==========
for idx, batch in enumerate(test_loader):
    input_tensor = batch['image'].to(device)
    name = batch['id'][0]

    with torch.no_grad():
        output, features, enc32_ = model(input_tensor, return_features=True)  # 返回中间特征

    save_path = os.path.join(save_dir, f"{name}_heatmap.jpg")
    visualize_feature_on_image(enc32_, input_tensor[0], save_path)
    print(f"Saved: {save_path}")

print("Feature map visualization completed.")
