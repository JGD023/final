import argparse
import math
import os
import random
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 导入自定义模块
from src.models.image_model import DMCI 
from src.datasets.image_dataset import ImageFolder 

# -----------------------------------------------------------
# 1. 损失函数 (修正 BPP 计算)
# -----------------------------------------------------------
class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}
        bpp_total = 0
        for name, likelihoods in output["likelihoods"].items():
            bpp = torch.sum(-torch.log2(torch.clamp(likelihoods, 1e-9, 1.0))) / num_pixels
            out[f"bpp_{name}"] = bpp
            bpp_total += bpp
        
        out["bpp_loss"] = bpp_total
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * (255 ** 2) * out["mse_loss"] + out["bpp_loss"]
        return out

# -----------------------------------------------------------
# 2. 训练与测试函数
# -----------------------------------------------------------
def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, rank, writer):
    model.train()
    train_dataloader.sampler.set_epoch(epoch)
    for i, d in enumerate(train_dataloader):
        d = d.to(rank)
        optimizer.zero_grad()
        qp_index = random.randint(0, model.module.get_qp_num() - 1)
        out_net = model(d, qp_index)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if rank == 0 and i % 50 == 0:
            step = epoch * len(train_dataloader) + i
            writer.add_scalar('Train/Loss', out_criterion["loss"].item(), step)
            writer.add_scalar('Train/Bpp', out_criterion["bpp_loss"].item(), step)
            writer.add_scalar('Train/MSE', out_criterion["mse_loss"].item(), step)
            print(f"Epoch {epoch} [{i}/{len(train_dataloader)}] | Loss: {out_criterion['loss'].item():.4f} | Bpp: {out_criterion['bpp_loss'].item():.4f} | QP: {qp_index}")

def test_epoch(epoch, test_dataloader, model, criterion, rank, writer):
    model.eval()
    device = torch.device(f"cuda:{rank}")
    mse_sum, bpp_sum, count = torch.zeros(1).to(device), torch.zeros(1).to(device), torch.zeros(1).to(device)
    
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            qp_index = model.module.get_qp_num() // 2 
            out_net = model(d, qp_index)
            out_criterion = criterion(out_net, d)
            mse_sum += out_criterion["mse_loss"]
            bpp_sum += out_criterion["bpp_loss"]
            count += 1

    dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(bpp_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)

    if rank == 0:
        avg_mse = mse_sum.item() / count.item()
        avg_bpp = bpp_sum.item() / count.item()
        psnr = 10 * math.log10(1 / (avg_mse + 1e-9))
        print(f"Test Epoch {epoch} | PSNR: {psnr:.2f} | Bpp: {avg_bpp:.4f} | Samples: {int(count.item())}")
        writer.add_scalar('Test/PSNR', psnr, epoch)
        writer.add_scalar('Test/Bpp', avg_bpp, epoch)

# -----------------------------------------------------------
# 3. DDP 工作进程
# -----------------------------------------------------------
def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = 'localhost', port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main_worker(rank, world_size, args):
    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)
    if rank != 0: sys.stdout = open(os.devnull, 'w') 

    writer = SummaryWriter(args.save_dir) if rank == 0 else None
    train_trans = transforms.Compose([transforms.RandomCrop((256, 256)), transforms.ToTensor()])
    test_trans = transforms.Compose([transforms.ToTensor()])

    train_set = ImageFolder(args.train_dataset, transform=train_trans, split="")
    test_set = ImageFolder(args.test_dataset, transform=test_trans, split="")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=DistributedSampler(train_set, world_size, rank), num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, sampler=DistributedSampler(test_set, world_size, rank, shuffle=False), num_workers=2)

    net = DMCI(N=256, z_channel=128).to(rank)
    net = DDP(net, device_ids=[rank], find_unused_parameters=False)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    criterion = RateDistortionLoss(lmbda=args.lambda_val)

    # --- 增强版续训加载逻辑 (解决 KeyError: 'state_dict') ---
    start_epoch = 0
    checkpoint_path = os.path.join(args.save_dir, "latest_checkpoint.pth.tar")
    if os.path.exists(checkpoint_path):
        if rank == 0: print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{rank}", weights_only=False)
        
        # 兼容性判断：检查是字典格式还是纯权重格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # 格式 A: 包含 epoch 和 optimizer 的新格式
            net.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            if rank == 0: print(f"Resuming from Epoch {start_epoch}")
        else:
            # 格式 B: 纯权重的旧格式 (如你 100 轮后的 31.90dB 模型)
            net.module.load_state_dict(checkpoint)
            if rank == 0: print("Old weights format detected. Starting from Epoch 0.")

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(net, criterion, train_loader, optimizer, epoch, rank, writer)
        test_epoch(epoch, test_loader, net, criterion, rank, writer)
        
        if rank == 0:
            state = {'epoch': epoch + 1, 'state_dict': net.module.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, checkpoint_path) 
            if (epoch + 1) % 10 == 0: 
                torch.save(state, os.path.join(args.save_dir, f"epoch_{epoch+1}.pth.tar"))

    if rank == 0: writer.close()
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='experiments/rt_v1')
    parser.add_argument('--lambda_val', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200) # 可以设大一点继续训
    parser.add_argument('--gpu_num', type=int, default=None)
    parser.add_argument('--port', type=str, default='12356')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir, exist_ok=True)
    world_size = args.gpu_num if args.gpu_num else torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args), join=True)

if __name__ == "__main__":
    main()