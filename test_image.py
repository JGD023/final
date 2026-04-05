import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 导入你的模型定义
from src.models.image_model import DMCI
from src.datasets.image_dataset import ImageFolder

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)

@torch.no_grad()
def evaluate_rd_curve(model, test_loader, device):
    model.eval()
    qp_num = model.get_qp_num()  # 应该是 8
    rd_results = []

    print(f"{'QP':<5} | {'PSNR (dB)':<10} | {'Bpp':<10}")
    print("-" * 30)

    for qp in range(qp_num):
        psnr_list = []
        bpp_list = []
        
        for i, d in enumerate(test_loader):
            d = d.to(device)
            # 获取模型输出
            out_net = model(d, qp)
            
            # 1. 计算 PSNR
            psnr = compute_psnr(d, torch.clamp(out_net["x_hat"], 0, 1))
            psnr_list.append(psnr)
            
            # 2. 计算 Bpp
            N, _, H, W = d.size()
            num_pixels = N * H * W
            bpp_total = 0
            for likelihoods in out_net["likelihoods"].values():
                bpp_total += torch.sum(-torch.log2(torch.clamp(likelihoods, 1e-9, 1.0))) / num_pixels
            bpp_list.append(bpp_total.item())

        avg_psnr = np.mean(psnr_list)
        avg_bpp = np.mean(bpp_list)
        rd_results.append({'qp': qp, 'psnr': avg_psnr, 'bpp': avg_bpp})
        
        print(f"{qp:<5} | {avg_psnr:<10.2f} | {avg_bpp:<10.4f}")

    return rd_results

def main():
    # 配置参数
    checkpoint_path = "./experiments/rt_v1/latest_checkpoint.pth.tar"
    test_dataset_path = "/data0/dataset/Kodak"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    net = DMCI(N=256, z_channel=128).to(device)
    
    # 处理 Checkpoint 格式兼容性
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        net.load_state_dict(checkpoint['state_dict'])
    else:
        net.load_state_dict(checkpoint)
    print(f"Loaded model from {checkpoint_path}")

    # 2. 准备数据集 (Kodak 通常不需要随机裁剪)
    test_transforms = transforms.Compose([transforms.ToTensor()])
    test_set = ImageFolder(test_dataset_path, transform=test_transforms, split="")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # 3. 运行评估
    results = evaluate_rd_curve(net, test_loader, device)

    # 4. 打印最终结果（方便直接复制到 Excel 或绘图）
    print("\nFinal RD Data for Plotting:")
    print("Bpp = " + str([round(r['bpp'], 4) for r in results]))
    print("PSNR = " + str([round(r['psnr'], 2) for r in results]))

if __name__ == "__main__":
    main()