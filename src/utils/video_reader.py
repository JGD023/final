# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image


class PNGReader():
    def __init__(self, src_path, width, height, start_num=1):
        self.eof = False
        self.src_path = src_path
        self.width = width
        self.height = height
        
        if not os.path.exists(self.src_path):
            raise FileNotFoundError(f"Directory not found: {self.src_path}")

        pngs = set(os.listdir(self.src_path))
        
        # 自动检测命名规则 (Prefix + Padding)
        if '001.png' in pngs:
            self.padding = 3
            self.prefix = ''
        elif '00001.png' in pngs:
            self.padding = 5
            self.prefix = ''
        elif 'im00001.png' in pngs:
            self.padding = 5
            self.prefix = 'im'
        elif 'f001.png' in pngs:
            self.padding = 3
            self.prefix = 'f'
        elif 'f1.png' in pngs:
            self.padding = 1
            self.prefix = 'f'
        else:
            # 打印出当前目录的前几个文件，帮助调试
            file_list = list(pngs)[:5]
            raise ValueError(f'Unknown image naming convention in {self.src_path}. Files found (sample): {file_list}')
            
        self.current_frame_index = start_num

    def read_one_frame(self):
        # rgb: 3xhxw uint8 numpy array
        if self.eof:
            return None

        # 根据检测到的前缀和填充生成文件名
        filename = f"{self.prefix}{str(self.current_frame_index).zfill(self.padding)}.png"
        png_path = os.path.join(self.src_path, filename)
                                
        if not os.path.exists(png_path):
            self.eof = True
            return None

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype(np.uint8).transpose(2, 0, 1)
        _, height, width = rgb.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return rgb

    def close(self):
        self.current_frame_index = 1

class YUV420Reader():
    def __init__(self, src_path, width, height, skip_frame=0):
        self.eof = False
        if not src_path.endswith('.yuv'):
            src_path = src_path + '.yuv'
        self.src_path = src_path

        self.y_size = width * height
        self.y_width = width
        self.y_height = height
        self.uv_size = width * height // 2
        self.uv_width = width // 2
        self.uv_height = height // 2
        # pylint: disable=R1732
        self.file = open(src_path, "rb")
        # pylint: enable=R1732
        skipped_frame = 0
        while not self.eof and skipped_frame < skip_frame:
            y = self.file.read(self.y_size)
            uv = self.file.read(self.uv_size)
            if not y or not uv:
                self.eof = True
            skipped_frame += 1

    def read_one_frame(self):
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        if self.eof:
            return None, None
        y = self.file.read(self.y_size)
        uv = self.file.read(self.uv_size)
        if not y or not uv:
            self.eof = True
            return None, None
        y = np.frombuffer(y, dtype=np.uint8).copy().reshape(1, self.y_height, self.y_width)
        uv = np.frombuffer(uv, dtype=np.uint8).copy().reshape(2, self.uv_height, self.uv_width)

        return y, uv

    def close(self):
        self.file.close()
