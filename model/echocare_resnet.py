import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from torchvision.models import resnet50, ResNet50_Weights


class AttentionPool2d(nn.Module):
    """Learnable attention-weighted spatial pooling. (B, C, H, W) -> (B, C)"""
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1, bias=False),
        )

    def forward(self, x):
        w = self.attn(x)
        w = w.flatten(2)
        w = F.softmax(w, dim=-1)
        x_flat = x.flatten(2)
        return torch.bmm(x_flat, w.transpose(1, 2)).squeeze(-1)


class ResNetUNETR_Seg(nn.Module):
    """ResNet-50 encoder + UNETR decoder with deep supervision and boundary head."""

    def __init__(self, seg_num_classes, in_chans=3, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        if in_chans != 3:
            backbone.conv1 = nn.Conv2d(in_chans, 64, 7, stride=2, padding=3, bias=False)

        self.pre_pool  = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.post_pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        sp, nm, dd = 2, "instance", 64
        self.encoder1  = UnetrBasicBlock(sp, in_chans, dd,     3, 1, nm, True)
        self.encoder2  = UnetrBasicBlock(sp, 64,       dd,     3, 1, nm, True)
        self.encoder3  = UnetrBasicBlock(sp, 256,    2 * dd,   3, 1, nm, True)
        self.encoder4  = UnetrBasicBlock(sp, 512,    4 * dd,   3, 1, nm, True)
        self.encoder5  = UnetrBasicBlock(sp, 1024,   8 * dd,   3, 1, nm, True)
        self.encoder10 = UnetrBasicBlock(sp, 2048,  16 * dd,   3, 1, nm, True)

        self.decoder5 = UnetrUpBlock(sp, 16 * dd, 8 * dd, 3, 2, nm, True)
        self.decoder4 = UnetrUpBlock(sp, 8 * dd,  4 * dd, 3, 2, nm, True)
        self.decoder3 = UnetrUpBlock(sp, 4 * dd,  2 * dd, 3, 2, nm, True)
        self.decoder2 = UnetrUpBlock(sp, 2 * dd,  1 * dd, 3, 2, nm, True)
        self.decoder1 = UnetrUpBlock(sp, 1 * dd,  1 * dd, 3, 2, nm, True)
        self.out      = UnetOutBlock(sp, dd, seg_num_classes)

        self.aux_out4 = nn.Conv2d(8 * dd, seg_num_classes, kernel_size=1)
        self.aux_out3 = nn.Conv2d(4 * dd, seg_num_classes, kernel_size=1)
        self.aux_out2 = nn.Conv2d(2 * dd, seg_num_classes, kernel_size=1)
        self.aux_out1 = nn.Conv2d(1 * dd, seg_num_classes, kernel_size=1)

        self.boundary_conv1 = nn.Sequential(
            nn.Conv2d(dd, dd, 3, padding=1, bias=False), nn.BatchNorm2d(dd), nn.ReLU(inplace=True))
        self.boundary_conv2 = nn.Sequential(
            nn.Conv2d(dd, dd // 2, 3, padding=1, bias=False), nn.BatchNorm2d(dd // 2), nn.ReLU(inplace=True))
        self.boundary_out = nn.Conv2d(dd // 2, 1, kernel_size=1)
        self.boundary_fuse = nn.Sequential(
            nn.Conv2d(dd + dd // 2, dd, 1, bias=False), nn.BatchNorm2d(dd), nn.ReLU(inplace=True))

        self.bottleneck_dim = 16 * dd

    def encode(self, x3):
        s_pre  = self.pre_pool(x3)
        s_post = self.post_pool(s_pre)
        s1 = self.layer1(s_post)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        enc0 = self.encoder1(x3)
        enc1 = self.encoder2(s_pre)
        enc2 = self.encoder3(s1)
        enc3 = self.encoder4(s2)
        enc4 = self.encoder5(s3)
        dec4 = self.encoder10(s4)
        return enc0, enc1, enc2, enc3, enc4, dec4

    def decode(self, enc0, enc1, enc2, enc3, enc4, dec4, return_aux=False):
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        bnd_feat = self.boundary_conv1(dec0)
        bnd_feat = self.boundary_conv2(bnd_feat)
        boundary_logits = self.boundary_out(bnd_feat)
        dec0_fused = self.boundary_fuse(torch.cat([dec0, bnd_feat], dim=1))
        out  = self.decoder1(dec0_fused, enc0)
        main = self.out(out)
        if not return_aux:
            return main
        aux4 = self.aux_out4(dec3)
        aux3 = self.aux_out3(dec2)
        aux2 = self.aux_out2(dec1)
        aux1 = self.aux_out1(dec0_fused)
        return main, aux4, aux3, aux2, aux1, boundary_logits


MORPH_TARGET_CLS = [2, 3, 4]  # AoHypo, AoV_sten, DORV
N_MORPH_TARGETS = len(MORPH_TARGET_CLS)
N_MORPH_FEATS = 20


class MorphHead(nn.Module):
    def __init__(self, n_feats=N_MORPH_FEATS, hidden=32,
                 n_targets=N_MORPH_TARGETS, dropout=0.2, gate_init=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feats, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, n_targets),
        )
        self.gate = nn.Parameter(torch.full((n_targets,), gate_init))

    def forward(self, morph_feats):
        return self.net(morph_feats) * self.gate.unsqueeze(0)


class Echocare_ResNet(nn.Module):
    """Single model for both phases.
    Phase 1: seg + cls jointly (use_morph=False)
    Phase 2: freeze seg, retrain cls + morph (use_morph=True)
    """

    def __init__(self, seg_class_num=15, cls_class_num=7, pretrained=True,
                 cls_hidden_dim=256, cls_dropout=0.7,
                 morph_hidden=32, morph_dropout=0.2, morph_gate_init=0.3):
        super().__init__()

        self.in_adapter = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.in_adapter.weight.zero_()
            self.in_adapter.weight[:, 0, 0, 0] = 1.0

        self.seg_net = ResNetUNETR_Seg(seg_num_classes=seg_class_num, in_chans=3, pretrained=pretrained)

        self.att_pool_enc3 = AttentionPool2d(256,  hidden_dim=64)
        self.att_pool_enc4 = AttentionPool2d(512,  hidden_dim=64)
        self.att_pool_dec4 = AttentionPool2d(1024, hidden_dim=128)

        self.cls_head = nn.Sequential(
            nn.Linear(1792, cls_hidden_dim), nn.BatchNorm1d(cls_hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, cls_class_num),
        )

        self.morph_head = MorphHead(n_feats=N_MORPH_FEATS, hidden=morph_hidden,
            n_targets=N_MORPH_TARGETS, dropout=morph_dropout, gate_init=morph_gate_init)
        self.use_morph = False
        self.seg_class_num = seg_class_num
        self.fp_dropout = nn.Dropout2d(p=0.5)
        self._seg_frozen = False

    def _compute_morph_features(self, seg_logits):
        soft = F.softmax(seg_logits, dim=1)
        areas = soft.mean(dim=(2, 3))
        area_fg = areas[:, 1:]
        eps = 1e-6
        LA, LV, RA, RV = areas[:, 1], areas[:, 2], areas[:, 3], areas[:, 4]
        Heart, Thorax = areas[:, 5], areas[:, 7]
        AscAo, MainPA = areas[:, 8], areas[:, 9]
        SVC, AoArch = areas[:, 12], areas[:, 13]
        ratios = torch.stack([
            LV / (LV + RV + eps), LA / (LA + RA + eps),
            Heart / (Thorax + eps), AscAo / (AscAo + MainPA + eps),
            AoArch / (AoArch + MainPA + eps), AoArch / (AoArch + SVC + eps),
        ], dim=1)
        return torch.cat([area_fg, ratios], dim=1)

    def _pool_multiscale(self, enc3, enc4, dec4):
        return torch.cat([self.att_pool_enc3(enc3), self.att_pool_enc4(enc4), self.att_pool_dec4(dec4)], dim=1)

    def _apply_morph(self, cls_logits, seg_logits):
        if not self.use_morph:
            return cls_logits
        morph_feats = self._compute_morph_features(seg_logits)
        morph_logits = self.morph_head(morph_feats)
        cls_logits = cls_logits.clone()
        for i, cidx in enumerate(MORPH_TARGET_CLS):
            cls_logits[:, cidx] = cls_logits[:, cidx] + morph_logits[:, i]
        return cls_logits

    def freeze_segmentation(self):
        for name, p in self.named_parameters():
            p.requires_grad = ("cls_head" in name or "att_pool" in name or "morph_head" in name)
        self._seg_frozen = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[freeze_segmentation] trainable={trainable/1e6:.4f}M / {total/1e6:.2f}M")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        self._seg_frozen = False

    def forward(self, x, need_fp=False, return_aux=False):
        x3 = self.in_adapter(x)
        enc0, enc1, enc2, enc3, enc4, dec4 = self.seg_net.encode(x3)

        if need_fp:
            p = [torch.cat([e, self.fp_dropout(e)], 0) for e in [enc0, enc1, enc2, enc3, enc4, dec4]]
            seg_outs = self.seg_net.decode(*p, return_aux=False)
            cls_outs = self.cls_head(self._pool_multiscale(p[3], p[4], p[5]))
            cls_outs = self._apply_morph(cls_outs, seg_outs)
            return seg_outs.chunk(2), cls_outs.chunk(2)

        if return_aux:
            main, aux4, aux3, aux2, aux1, bnd = self.seg_net.decode(
                enc0, enc1, enc2, enc3, enc4, dec4, return_aux=True)
            cls_logits = self._apply_morph(self.cls_head(self._pool_multiscale(enc3, enc4, dec4)), main)
            return (main, aux4, aux3, aux2, aux1, bnd), cls_logits

        seg_logits = self.seg_net.decode(enc0, enc1, enc2, enc3, enc4, dec4, return_aux=False)
        cls_logits = self._apply_morph(self.cls_head(self._pool_multiscale(enc3, enc4, dec4)), seg_logits)
        return seg_logits, cls_logits

    def forward_cls_only(self, x):
        x3 = self.in_adapter(x)
        with torch.no_grad():
            enc0, enc1, enc2, enc3, enc4, dec4 = self.seg_net.encode(x3)
        feats = self._pool_multiscale(enc3, enc4, dec4)
        cls_logits = self.cls_head(feats)
        if self.use_morph:
            with torch.no_grad():
                seg_logits = self.seg_net.decode(enc0, enc1, enc2, enc3, enc4, dec4, return_aux=False)
            morph_feats = self._compute_morph_features(seg_logits)
            morph_logits = self.morph_head(morph_feats)
            cls_logits = cls_logits.clone()
            for i, cidx in enumerate(MORPH_TARGET_CLS):
                cls_logits[:, cidx] = cls_logits[:, cidx] + morph_logits[:, i]
        return cls_logits
