"""
Swin-Transformer OCR (문자 인식) 모델.

원본: AI Hub 공공 OCR 데이터 — swin_transformer/models.py (MIT License)

CRAFT 가 검출한 단어 영역 이미지를 받아 텍스트를 반환하는 디코더.
Encoder: timm SwinTransformer → Decoder: x_transformers 오토리그레시브.
"""
from __future__ import annotations

from contextlib import nullcontext as _nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as tf

from timm.models.swin_transformer import SwinTransformer
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import (
    AutoregressiveWrapper,
    top_k,
    top_p,
)

try:
    from entmax import entmax15 as entmax
    ENTMAX_ALPHA = 1.5
except ImportError:
    entmax = None
    ENTMAX_ALPHA = 1.5


class _CustomSwinTransformer(SwinTransformer):
    def __init__(self, img_size=224, *args, **kwargs):
        super().__init__(img_size=img_size, *args, **kwargs)
        self.height, self.width = img_size

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)
        return x


class _CustomARWrapper(AutoregressiveWrapper):
    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_logits_fn=top_k,
        filter_thres=0.9,
        **kwargs,
    ):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)
        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape
        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
            elif entmax is not None and filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)
            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]
        if num_dims == 1:
            out = out.squeeze(0)
        self.net.train(was_training)
        return out


class SwinTransformerOCR(nn.Module):
    """
    Swin-Transformer 기반 문자 인식 모델.

    predict(images: list[PIL.Image]) → list[str]
    """

    def __init__(
        self,
        tokenizer,
        *,
        width: int = 448,
        height: int = 112,
        channels: int = 1,
        patch_size: int = 4,
        window_size: int = 7,
        encoder_dim: int = 96,
        encoder_depth: list[int] | None = None,
        encoder_heads: list[int] | None = None,
        max_seq_len: int = 32,
        decoder_dim: int = 384,
        decoder_heads: int = 8,
        decoder_depth: int = 4,
        decoder_cfg: dict | None = None,
        temperature: float = 0.2,
        pad_token: int = 0,
        bos_token: int = 1,
        eos_token: int = 2,
    ):
        super().__init__()
        if encoder_depth is None:
            encoder_depth = [2, 6, 2]
        if encoder_heads is None:
            encoder_heads = [6, 12, 24]
        if decoder_cfg is None:
            decoder_cfg = {
                "cross_attend": True,
                "ff_glu": False,
                "attn_on_attn": False,
                "use_scalenorm": False,
                "rel_pos_bias": False,
            }

        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self._height = height
        self._width = width

        self.encoder = _CustomSwinTransformer(
            img_size=(height, width),
            patch_size=patch_size,
            in_chans=channels,
            num_classes=0,
            window_size=window_size,
            embed_dim=encoder_dim,
            depths=encoder_depth,
            num_heads=encoder_heads,
        )
        self.decoder = _CustomARWrapper(
            TransformerWrapper(
                num_tokens=len(tokenizer),
                max_seq_len=max_seq_len,
                attn_layers=Decoder(
                    dim=decoder_dim,
                    depth=decoder_depth,
                    heads=decoder_heads,
                    **decoder_cfg,
                ),
            ),
            pad_value=pad_token,
        )
        self.transform = tf.Compose([
            tf.Resize((height, width)),
            tf.Grayscale(),
            tf.ToTensor(),
        ])

    def forward(self, x):
        encoded = self.encoder(x)
        dec = self.decoder.generate(
            torch.LongTensor([self.bos_token] * len(x))[:, None].to(x.device),
            self.max_seq_len,
            eos_token=self.eos_token,
            context=encoded,
            temperature=self.temperature,
        )
        return dec

    @torch.no_grad()
    def predict(self, images: list, batch_size: int = 64) -> list[str]:
        if not images:
            return [""]
        result: list[str] = []
        device = next(self.parameters()).device

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            batch = []
            for image in batch_imgs:
                try:
                    batch.append(self.transform(image))
                except Exception:
                    batch.append(torch.ones((1, self._height, self._width)) * 255)

            use_amp = device.type == "cuda"
            ctx = torch.amp.autocast("cuda") if use_amp else _nullcontext()
            with ctx:
                stacked = torch.stack(batch).to(device)
                dec = self(stacked)
            result += self.tokenizer.decode(dec)

        return result
