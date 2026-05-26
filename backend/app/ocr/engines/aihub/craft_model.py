"""
CRAFT (Character Region Awareness for Text Detection) 모델.

원본: AI Hub 공공 OCR 데이터 — craft/model.py + craft/craft_modules.py
라이선스: MIT

VGG16-BN 백본 + U-Net 디코더로 문자/어피니티 히트맵 생성.
predict() 에서 이미지 → 단어 바운딩 박스 리스트를 반환.
"""
from __future__ import annotations

import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image
from collections import namedtuple

from contextlib import nullcontext as _nullcontext
from app.ocr.engines.aihub.craft_detect import generate_word_bbox_batch, merge_adjacent_boxes


class _VGG16BN(nn.Module):
    """VGG16-BN 특징 추출기 (fc6/fc7 변형 포함)."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=None).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(12)])
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(12, 19)])
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(19, 29)])
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(29, 39)])
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )
        _init_weights(self.slice5.modules())

    def forward(self, x):
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        VggOut = namedtuple("VggOut", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])
        return VggOut(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def _init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class CRAFTModel(nn.Module):
    """
    CRAFT 텍스트 검출 네트워크.

    forward() → (score_map [B,H,W,2], feature)
    predict(np_image) → list[[lx, ly, rx, ry], ...]  단어 바운딩 박스
    """

    def __init__(self, img_size: int = 1536,
                 threshold_character: float = 0.6,
                 threshold_affinity: float = 0.3,
                 threshold_word: float = 0.7):
        super().__init__()
        self.img_size = img_size
        self.threshold_character = threshold_character
        self.threshold_affinity = threshold_affinity
        self.threshold_word = threshold_word

        self.basenet = _VGG16BN()
        self.upconv1 = _DoubleConv(1024, 512, 256)
        self.upconv2 = _DoubleConv(512, 256, 128)
        self.upconv3 = _DoubleConv(256, 128, 64)
        self.upconv4 = _DoubleConv(128, 64, 32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )
        self.transform = transforms.ToTensor()
        self.eval()

    def forward(self, x):
        sources = self.basenet(x)

        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        y = F.interpolate(y, size=sources[2].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        y = F.interpolate(y, size=sources[3].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)
        y = F.interpolate(y, size=sources[4].size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)
        y = self.conv_cls(feature)
        return y.permute(0, 2, 3, 1), feature

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> list[list[int]]:
        """
        단일 이미지에서 단어 바운딩 박스를 검출.

        Args:
            image: BGR np.ndarray [H, W, C]
        Returns:
            list of [lx, ly, rx, ry] (정수 좌표)
        """
        img_pil, (h_pad, w_pad), ratio = self._resize(image, self.img_size)
        device = next(self.parameters()).device
        use_amp = device.type == "cuda"

        ctx = torch.amp.autocast("cuda") if use_amp else _nullcontext()
        with ctx:
            x = self.transform(img_pil).unsqueeze(0).to(device)
            out, _ = self.forward(x)

        out = out.float()
        half = self.img_size // 4
        character_map = out[:, :, :, 0].detach().cpu().numpy()
        affinity_map = out[:, :, :, 1].detach().cpu().numpy()

        characters = [
            character_map[:, :half, :half],
            character_map[:, :half, half:],
            character_map[:, half:, :half],
            character_map[:, half:, half:],
        ]
        affinities = [
            affinity_map[:, :half, :half],
            affinity_map[:, :half, half:],
            affinity_map[:, half:, :half],
            affinity_map[:, half:, half:],
        ]

        total_boxes: list = []
        merged_boxes = np.empty((0, 2, 2))

        for i, (character, affinity) in enumerate(zip(characters, affinities)):
            boxes = generate_word_bbox_batch(
                character, affinity,
                character_threshold=self.threshold_character,
                affinity_threshold=self.threshold_affinity,
                word_threshold=self.threshold_word,
            )
            boxes = boxes[0][:, [0, 2], :, :].squeeze(2)

            if i == 1:
                boxes[:, :, 0] += half
                boxes, total_boxes[0], nb = merge_adjacent_boxes(
                    boxes, total_boxes[0], affinity_map[0, :, :], "left")
                merged_boxes = np.concatenate((merged_boxes, nb))
            elif i == 2:
                boxes[:, :, 1] += half
                boxes, total_boxes[0], nb = merge_adjacent_boxes(
                    boxes, total_boxes[0], affinity_map[0, :, :], "up")
                merged_boxes = np.concatenate((merged_boxes, nb))
            elif i == 3:
                boxes[:, :, 0] += half
                boxes[:, :, 1] += half
                boxes, total_boxes[1], nb = merge_adjacent_boxes(
                    boxes, total_boxes[1], affinity_map[0, :, :], "up")
                merged_boxes = np.concatenate((merged_boxes, nb))
                boxes, total_boxes[2], nb = merge_adjacent_boxes(
                    boxes, total_boxes[2], affinity_map[0, :, :], "left")
                merged_boxes = np.concatenate((merged_boxes, nb))

            total_boxes.append(boxes)

        all_boxes = np.concatenate(total_boxes + [merged_boxes]) * 2
        all_boxes[:, :, 0] -= w_pad
        all_boxes[:, :, 1] -= h_pad
        all_boxes = all_boxes.reshape(-1, 4) / ratio

        return all_boxes.astype(int).tolist()

    @staticmethod
    def _resize(image: np.ndarray, big_side: int = 1536):
        height, width, _ = image.shape
        ratio = big_side / max(height, width)
        new_size = (int(width * ratio), int(height * ratio))
        image = cv2.resize(image, new_size)

        big_image = np.ones([big_side, big_side, 3], dtype=np.float32) * 255
        h_pad = (big_side - image.shape[0]) // 2
        w_pad = (big_side - image.shape[1]) // 2
        big_image[h_pad:h_pad + image.shape[0], w_pad:w_pad + image.shape[1]] = image
        big_image = big_image.astype(np.uint8)
        return Image.fromarray(big_image), (h_pad, w_pad), ratio
