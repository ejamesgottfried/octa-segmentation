"""FlexUNet: a single U-Net implementation with toggleable attention,
residual, and squeeze-and-excitation (SE) components, so an architecture
ablation can compare them against a byte-identical backbone.

See MODEL_REGISTRY at the bottom for the named variants actually used in
the ablation scripts, and build_model() as the single entry point that
train.py / test_model.py call by name.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Building blocks ----

class DoubleConv(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU blocks. The plain-U-Net backbone unit."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018).

    Gates an encoder skip `x` using a decoder gating signal `g` at the same
    spatial resolution. Returns the skip re-weighted by a per-pixel coefficient
    in [0, 1].
    """
    def __init__(self, gating_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        attn = self.relu(self.W_g(g) + self.W_x(x))
        attn = self.psi(attn)
        return x * attn


class ConvBlock(nn.Module):
    """Backbone block for FlexUNet.

    With residual=False and se=False this is exactly DoubleConv (no extra
    params), which is what makes FlexUNet(all off) identical to the plain UNet.
    Flags add SE recalibration and/or a residual shortcut on the branch output
    (SE-ResNet ordering: SE on the branch, then add shortcut).
    """
    def __init__(self, in_channels, out_channels, residual=False, se=False, se_reduction=16):
        super().__init__()
        self.residual = residual
        self.conv = DoubleConv(in_channels, out_channels)
        self.se = SEBlock(out_channels, se_reduction) if se else None
        if residual:
            self.shortcut = (nn.Identity() if in_channels == out_channels
                             else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.se is not None:
            out = self.se(out)
        if self.residual:
            out = self.out_act(out + self.shortcut(x))
        return out


# ---- Flexible U-Net ----

class FlexUNet(nn.Module):
    """U-Net with toggleable components for clean architectural ablation.

    Every flag is one variable held against a byte-identical backbone, so any
    metric delta is attributable to that flag alone.
      attention : additive attention gates on every skip connection
      residual  : residual shortcut inside each conv block
      se        : squeeze-and-excitation channel attention inside each conv block

    FlexUNet(attention=False, residual=False, se=False) == the plain UNet.
    Output is raw logits, (B, num_classes, H, W). Apply sigmoid at metric time.

    Note: deep supervision is deliberately NOT a flag here, since it changes
    forward() to return multiple outputs and would force loss/train changes.
    Add it as a separate path later if wanted.
    """
    def __init__(self, in_channels=1, num_classes=1,
                 features=(64, 128, 256, 512),
                 attention=False, residual=False, se=False, se_reduction=16):
        super().__init__()
        f1, f2, f3, f4 = features
        self.attention = attention

        def block(i, o):
            return ConvBlock(i, o, residual=residual, se=se, se_reduction=se_reduction)

        # Encoder
        self.enc1 = block(in_channels, f1)
        self.enc2 = block(f1, f2)
        self.enc3 = block(f2, f3)
        self.enc4 = block(f3, f4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = block(f4, f4 * 2)

        # Decoder (deepest first, matching the plain UNet's up4..up1 order)
        self.up4 = nn.ConvTranspose2d(f4 * 2, f4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(f4, f4, f4 // 2) if attention else None
        self.dec4 = block(f4 * 2, f4)

        self.up3 = nn.ConvTranspose2d(f4, f3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(f3, f3, f3 // 2) if attention else None
        self.dec3 = block(f3 * 2, f3)

        self.up2 = nn.ConvTranspose2d(f3, f2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(f2, f2, f2 // 2) if attention else None
        self.dec2 = block(f2 * 2, f2)

        self.up1 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(f1, f1, f1 // 2) if attention else None
        self.dec1 = block(f1 * 2, f1)

        self.final = nn.Conv2d(f1, num_classes, kernel_size=1)

    def _merge(self, gate, decoder_feat, skip_feat):
        # Gate the skip if attention is on, then concat decoder-first (UNet order).
        if gate is not None:
            skip_feat = gate(decoder_feat, skip_feat)
        return torch.cat([decoder_feat, skip_feat], dim=1)

    def forward(self, x):
        # Encoder (save each level for skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + (optionally gated) skip connections
        d4 = self.up4(b)
        d4 = self.dec4(self._merge(self.att4, d4, e4))

        d3 = self.up3(d4)
        d3 = self.dec3(self._merge(self.att3, d3, e3))

        d2 = self.up2(d3)
        d2 = self.dec2(self._merge(self.att2, d2, e2))

        d1 = self.up1(d2)
        d1 = self.dec1(self._merge(self.att1, d1, e1))

        return self.final(d1)  # logits


# ---- Registry: named configs so train.py / test_model.py just pass a name ----

MODEL_REGISTRY = {
    "unet":           lambda **kw: FlexUNet(attention=False, residual=False, se=False, **kw),
    "attention_unet": lambda **kw: FlexUNet(attention=True,  residual=False, se=False, **kw),
    "res_unet":       lambda **kw: FlexUNet(attention=False, residual=True,  se=False, **kw),
    "se_unet":        lambda **kw: FlexUNet(attention=False, residual=False, se=True,  **kw),
    "att_res_unet":   lambda **kw: FlexUNet(attention=True,  residual=True,  se=False, **kw),
}


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)