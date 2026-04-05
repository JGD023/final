import math

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from ..layers.cuda_inference import build_index_dec, build_index_enc, process_with_mask


class EntropyCoder():
    def __init__(self):
        super().__init__()

        from MLCodec_extensions_cpp import RansEncoder, RansDecoder
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from MLCodec_extensions_cpp import pmf_to_quantized_cdf as _pmf_to_cdf
        cdf = _pmf_to_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def reset(self):
        self.encoder.reset()

    def add_cdf(self, cdf, cdf_length, offset):
        enc_cdf_idx = self.encoder.add_cdf(cdf, cdf_length, offset)
        dec_cdf_idx = self.decoder.add_cdf(cdf, cdf_length, offset)
        assert enc_cdf_idx == dec_cdf_idx
        return enc_cdf_idx

    def encode_y(self, symbols, cdf_group_index):
        # symbols: int16, high 8 bits: int8 symbol to be encoded; low 8 bits: uint8 index to use
        assert symbols.dtype == torch.int16
        self.encoder.encode_y(symbols.cpu().numpy(), cdf_group_index)

    def encode_z(self, symbols, cdf_group_index, start_offset, per_channel_size):
        self.encoder.encode_z(symbols.to(torch.int8).cpu().numpy(),
                              cdf_group_index, start_offset, per_channel_size)

    def flush(self):
        self.encoder.flush()

    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))

    def decode_y(self, indexes, cdf_group_index):
        self.decoder.decode_y(indexes.to(torch.uint8).cpu().numpy(), cdf_group_index)

    def decode_and_get_y(self, indexes, cdf_group_index, device, dtype):
        rv = self.decoder.decode_and_get_y(indexes.to(torch.uint8).cpu().numpy(), cdf_group_index)
        rv = torch.as_tensor(rv)
        return rv.to(device).to(dtype)

    def decode_z(self, total_size, cdf_group_index, start_offset, per_channel_size):
        self.decoder.decode_z(total_size, cdf_group_index, start_offset, per_channel_size)

    def get_decoded_tensor(self, device, dtype, non_blocking=False):
        rv = self.decoder.get_decoded_tensor()
        rv = torch.as_tensor(rv)
        return rv.to(device, non_blocking=non_blocking).to(dtype)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.encoder.set_use_two_encoders(use_two_entropy_coders)
        self.decoder.set_use_two_decoders(use_two_entropy_coders)


class Bitparm(nn.Module):
    def __init__(self, qp_num, channel, final=False):
        super().__init__()
        self.final = final #标记是否为 BitEstimator 中的最后一层变换

        # 定义可学习参数 h (控制缩放/斜率)
        # 形状为 [qp_num, channel, 1, 1]，为每个质量等级和每个通道存储独立参数
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        
        # 定义可学习参数 b (控制位移/截距)
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        
        # 如果不是最后一层，额外定义参数 a (用于非线性混合)
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        else:
            self.a = None

    def forward(self, x, index):
        # 1. 记录输入的原始形状 (例如训练时的 [B, C, H, W] 或推理时的 [N, 1])
        origin_shape = x.shape
        
        # 2. 核心修改：展平数据以适配 index_select
        # index_select 的 index 参数必须是 1-D vector
        x = x.reshape(-1, 1)
        index = index.reshape(-1).long() # 确保索引是长整型

        # 3. 将参数矩阵也展平为 [M, 1] 形式进行索引选择
        # self.h.reshape(-1, 1) 会将 [qp_num, channel, 1, 1] 展平
        h = torch.index_select(self.h.reshape(-1, 1), 0, index)
        b = torch.index_select(self.b.reshape(-1, 1), 0, index)
        
        # 4. 执行仿射变换
        # 使用 F.softplus 确保 scale 参数大于 0
        x = x * F.softplus(h) + b
        
        if self.final:
            # 5. 返回前必须还原回原始形状，保证后续网络维度正确
            return x.reshape(origin_shape)

        # 6. 非线性变换层处理
        a = torch.index_select(self.a.reshape(-1, 1), 0, index)
        x = x + torch.tanh(x) * torch.tanh(a)
        
        # 7. 还原形状并返回
        return x.reshape(origin_shape)


class AEHelper():
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self.cdf_group_index = None
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset


class BitEstimator(AEHelper, nn.Module):
    def __init__(self, qp_num, channel):
        super().__init__()
        self.f1 = Bitparm(qp_num, channel)
        self.f2 = Bitparm(qp_num, channel)
        self.f3 = Bitparm(qp_num, channel)
        self.f4 = Bitparm(qp_num, channel, True)
        self.qp_num = qp_num
        self.channel = channel

    def forward(self, x, index):
        return self.get_cdf(x, index)

    def get_logits_cdf(self, x, index):
        x = self.f1(x, index)
        x = self.f2(x, index)
        x = self.f3(x, index)
        x = self.f4(x, index)
        return x

    def get_cdf(self, x, index):
        return torch.sigmoid(self.get_logits_cdf(x, index))

    def update(self, entropy_coder):
        self.entropy_coder = entropy_coder

        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
            index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

            minima = medians + 8
            for i in range(8, 1, -1):
                samples = torch.zeros_like(medians) - i
                probs = self.forward(samples, index)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 8
            for i in range(8, 1, -1):
                samples = torch.zeros_like(medians) + i
                probs = self.forward(samples, index)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)

            samples = samples[None, None, None, :] + pmf_start

            half = float(0.5)

            lower = self.forward(samples - half, index)
            upper = self.forward(samples + half, index)
            pmf = upper - lower

            pmf = pmf[:, :, 0, :]
            upper = self.forward(maxima.to(torch.float32), index)
            tail_mass = lower[:, :, 0, :1] + (1.0 - upper[:, :, 0, -1:])

            pmf = pmf.reshape([-1, max_length])
            tail_mass = tail_mass.reshape([-1, 1])
            pmf_length = pmf_length.reshape([-1])
            offset = offset.reshape([-1])
            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)
            self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def build_indexes(self, size, qp):
        B, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1) + qp * self.channel
        return indexes.repeat(B, 1, H, W)

    def encode_z(self, x, qp):
        _, _, H, W = x.size()
        return self.entropy_coder.encode_z(x.reshape(-1), self.cdf_group_index, qp * self.channel,
                                           H * W)

    def decode_z(self, size, qp):
        self.entropy_coder.decode_z(self.channel * size[0] * size[1], self.cdf_group_index,
                                    qp * self.channel, size[0] * size[1])

    def get_z(self, size, device, dtype):
        output_size = (1, self.channel, size[0], size[1])
        val = self.entropy_coder.get_decoded_tensor(device, dtype, non_blocking=True)
        return val.reshape(output_size)
    
    def get_likelihood(self, x, qp_index):
    # x: 输入的超先验特征 z (float32)
    # qp_index: 当前使用的量化参数索引
    
        half = 0.5
    
    # 1. 构建索引 (确保每个通道使用正确的分布参数)
    # size: [B, C, H, W]
        indexes = self.build_indexes(x.size(), qp_index).to(x.device)
    
    # 2. 计算上限和下限的 CDF 值
    # 使用 get_cdf 或直接使用 sigmoid(get_logits_cdf)
        f_upper = self.get_cdf(x + half, indexes)
        f_lower = self.get_cdf(x - half, indexes)
    
    # 3. 计算似然值 (概率质量)
        likelihood = f_upper - f_lower
    
    # 4. 边界处理 (防止由于浮点误差导致的负数或零，避免 log 运算崩溃)
        likelihood = torch.clamp(likelihood, min=1e-9)
    
        return likelihood


class GaussianEncoder(AEHelper,nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_min = 0.11
        self.scale_max = 16.0
        self.scale_level = 128  # <= 256
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)
        self.log_step_recip = 1. / self.log_scale_step

        self.force_zero_thres = None
        self.decode_index_cache = {}
        self.decode_zeros_cache = {}

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

    def update(self, entropy_coder, force_zero_thres=None):
        self.entropy_coder = entropy_coder
        self.force_zero_thres = force_zero_thres

        pmf_center = torch.zeros_like(self.scale_table) + 8
        scales = torch.zeros_like(pmf_center) + self.scale_table
        cdf_distribution = torch.distributions.normal.Normal(0., scales)
        for i in range(8, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        cdf_distribution = torch.distributions.normal.Normal(0., scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2, -pmf_center)
        self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def process_with_mask(self, y, scales, means, mask):
        return process_with_mask(y, scales, means, mask, self.force_zero_thres)

    def build_indexes_decoder(self, scales):
        scales = scales.reshape(-1)
        indexes, skip_cond = build_index_dec(scales, self.scale_min, self.scale_max,
                                             self.log_scale_min, self.log_step_recip,
                                             self.force_zero_thres)
        if self.force_zero_thres is not None:
            indexes = indexes[skip_cond]
        return indexes, skip_cond

    def build_indexes_encoder(self, symbols, scales):
        symbols = symbols.reshape(-1)
        scales = scales.reshape(-1)
        symbols = build_index_enc(symbols, scales, self.scale_min, self.scale_max,
                                  self.log_scale_min, self.log_step_recip, self.force_zero_thres)
        return symbols

    def encode_y(self, x, scales):
        symbols = self.build_indexes_encoder(x, scales)
        return self.entropy_coder.encode_y(symbols, self.cdf_group_index)

    def get_decode_index_cache(self, num, device):
        if num not in self.decode_index_cache:
            c = torch.arange(0, num, dtype=torch.int32, device=device)
            self.decode_index_cache[num] = c

        return self.decode_index_cache[num]

    def get_decode_zeros_cache(self, num, device):
        if num not in self.decode_zeros_cache:
            c = torch.zeros(num, dtype=torch.int32, device=device)
            self.decode_zeros_cache[num] = c

        return self.decode_zeros_cache[num].clone()

    def decode_and_get_y(self, scales, dtype, device):
        indexes, skip_cond = self.build_indexes_decoder(scales)
        self.decode_y(indexes)
        return self.get_y(scales.shape, scales.numel(), dtype, device, skip_cond, indexes)

    def decode_y(self, indexes):
        self.entropy_coder.decode_y(indexes, self.cdf_group_index)

    def get_y(self, shape, numel, dtype, device, skip_cond, indexes):
        if len(indexes) == 0:
            return torch.zeros(shape, dtype=dtype, device=device)
        if skip_cond is not None:
            curr_index = self.get_decode_index_cache(numel, device)
            back_index = self.get_decode_zeros_cache(numel, device)
            back_index.masked_scatter_(skip_cond, curr_index)
        val = self.entropy_coder.get_decoded_tensor(device, dtype, non_blocking=True)
        if skip_cond is not None:
            y = torch.index_select(val, 0, back_index) * skip_cond
            return y.reshape(shape)
        return val.reshape(shape)
    
    def forward(self, y, scales, means):
        """
        用于训练模式下的前向传递：
        1. 模拟量化 (加噪声)
        2. 计算似然值 (Likelihoods)
        """
        half = 0.5
        
        # 1. 模拟量化：训练时加噪声，推理时取整
        if self.training:
            # 使用均匀噪声模拟量化误差，确保梯度可传播
            noise = torch.nn.init.uniform_(torch.zeros_like(y), -half, half)
            y_hat = y + noise
        else:
            # 推理阶段使用硬量化
            y_hat = torch.round(y)

        # 2. 计算似然值 (Likelihood)
        # 概率 P = CDF(y_hat + 0.5) - CDF(y_hat - 0.5)
        # 这里使用 torch.distributions 提供的 Normal 模块
        
        # 确保 scales 不会过小导致数值不稳定 (通常在 1e-6 以上)
        scales = torch.clamp(scales, min=self.scale_min)
        
        dist = torch.distributions.normal.Normal(means, scales)
        
        # 计算上限和下限的累积分布
        upper = dist.cdf(y_hat + half)
        lower = dist.cdf(y_hat - half)
        
        likelihood = upper - lower
        
        # 3. 数值安全处理：防止出现 0 导致后续 log(0) 产生 NaN
        likelihood = torch.clamp(likelihood, min=1e-9)
        
        return y_hat, likelihood
