
import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F
class _CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # use resnet50 final fc for classification
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base
    def forward(self, x):
        return self.model(x)

class _ViT(nn.Module):
    def __init__(self, num_classes, proj_dim=384):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        feat_dim = vit.heads.head.in_features  # should be 768
        vit.heads = nn.Identity()
        self.vit = vit
        self.proj = nn.Linear(feat_dim, proj_dim)
        # local token mixer
        self.local = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_feat = self.vit(x)
        if vit_feat.ndim == 4:
            vit_feat = vit_feat.flatten(1)
        v = self.proj(vit_feat)  # now input dimension matches 768 → 384
        l = self.local(x)
        fused = torch.cat([v, l], dim=1)
        return self.classifier(fused)

class CNN_ViT_Fusion(nn.Module):
    def __init__(self, num_classes, res_proj=512, vit_proj=384, q_dim=128, use_small=False):
        super().__init__()
        # resnet backbone without classifier
        if use_small:
            res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.res_backbone = nn.Sequential(*list(res.children())[:-1])
        res_feat_dim = res.fc.in_features
        self.res_proj = nn.Linear(res_feat_dim, res_proj)

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit = vit
        vit_feat_dim = vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        self.vit_proj = nn.Linear(vit_feat_dim, vit_proj)

        # quantum-inspired mapping (simple)
        self.q_fc = nn.Linear(vit_proj, q_dim)
        fused_dim = res_proj + vit_proj + q_dim
        self.classifier = nn.Sequential(nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x):
        vit_feat = self.vit(x)
        if vit_feat.ndim == 4:
            vit_feat = vit_feat.flatten(1)
        vit_emb = self.vit_proj(vit_feat)
        res_feat = self.res_backbone(x).flatten(1)
        res_emb = self.res_proj(res_feat)
        q = torch.tanh(self.q_fc(vit_emb)) * F.softmax(self.q_fc(vit_emb), dim=-1)
        fused = torch.cat([res_emb, vit_emb, q], dim=1)
        return self.classifier(fused)

class EnhancedCNN(nn.Module):
    def __init__(self, num_classes, proj_dim=512, small=False):
        super().__init__()
        if small:
            res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(res.children())[:-1])
        self.proj = nn.Sequential(nn.Linear(res.fc.in_features, proj_dim), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Linear(proj_dim, num_classes)
    def forward(self, x):
        f = self.backbone(x).flatten(1)
        p = self.proj(f)
        return self.classifier(p)

class ImprovedViT(nn.Module):
    def __init__(self, num_classes, proj_dim=384):
        super().__init__()
        # Load ViT backbone
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        feat_dim = vit.heads.head.in_features  # Get correct output dimension (usually 768)
        vit.heads = nn.Identity()
        self.vit = vit

        # Projection to desired dimension
        self.proj = nn.Linear(feat_dim, proj_dim)

        # Local feature extractor
        self.local = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # Classifier with fusion
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_feat = self.vit(x)  # (B, 768) for vit_b_16
        if vit_feat.ndim == 4:  # If ViT returned (B, C, 1, 1)
            vit_feat = vit_feat.flatten(1)

        v = self.proj(vit_feat)  # Now 768 → 384 works
        l = self.local(x)
        fused = torch.cat([v, l], dim=1)
        return self.classifier(fused)

# Proposed MediFlora-Net
class MediFloraNet(nn.Module):
    def __init__(self, num_classes, res_out=512, vit_out=384, q_out=128, small=False, pretrained=True):
        super().__init__()
        # ResNet backbone
        if small:
            res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.res_backbone = nn.Sequential(*list(res.children())[:-1])
        self.res_proj = nn.Linear(res.fc.in_features, res_out)
        # ViT
        try:
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = vit.heads.head.in_features
            vit.heads = nn.Identity()
            self.vit_backbone = vit
            self.vit_proj = nn.Linear(feat_dim, vit_out)
        except Exception:
            # fallback small embed
            self.vit_backbone = nn.Sequential(nn.Conv2d(3, 64, kernel_size=16, stride=16), nn.AdaptiveAvgPool2d(1))
            self.vit_proj = nn.Linear(64, vit_out)
        self.q_fc = nn.Linear(vit_out, q_out)
        fused_dim = res_out + vit_out + q_out
        self.classifier = nn.Sequential(nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x):
        vit_feat = self.vit_backbone(x)
        if isinstance(vit_feat, torch.Tensor) and vit_feat.ndim == 4:
            vit_feat = vit_feat.flatten(1)
        vit_emb = self.vit_proj(vit_feat)
        res_feat = self.res_backbone(x).flatten(1)
        res_emb = self.res_proj(res_feat)
        q = torch.tanh(self.q_fc(vit_emb))
        q = q * F.softmax(q, dim=-1)
        fused = torch.cat([res_emb, vit_emb, q], dim=1)
        return self.classifier(fused)

class ViT_ReT_(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        vit.heads = nn.Identity()
        feat_dim = vit.heads.head.in_features if hasattr(vit.heads,'head') else 768
        self.vit = vit
        self.reproj = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU(), nn.Linear(512, num_classes))
    def forward(self, x):
        v = self.vit(x)
        if v.ndim == 4:
            v = v.flatten(1)
        return self.reproj(v)

def get_swin_(num_classes):
    try:
        swin = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        swin.head = nn.Linear(swin.head.in_features, num_classes)
        return swin
    except Exception:
        return _CNN(num_classes)

class BrainNPT_(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(res.children())[:-1])
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=1)
        self.fc = nn.Linear(res.fc.in_features, num_classes)
    def forward(self, x):
        f = self.backbone(x).flatten(1)
        # create dummy sequence: repeat feature -> seq_len 4
        seq = f.unsqueeze(1).repeat(1,4,1)  # Bx4xD
        seq = seq.transpose(0,1)  # 4xBxD
        t = self.trans(seq).mean(dim=0)
        return self.fc(t)

class iEEG_HCT_(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(res.children())[:-2])  # B × 512 × H × W
        self.pool = nn.AdaptiveAvgPool2d((7, 7))                   # → 7×7 = 49 tokens
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=False),
            num_layers=1
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        f = self.backbone(x)                   # B × 512 × H × W
        p = self.pool(f).flatten(2)            # B × 512 × 49
        seq = p.transpose(1, 2)                # B × 49 × 512
        seq = seq.transpose(0, 1)              # 49 × B × 512

        t = self.trans(seq).mean(dim=0)        # B × 512
        out = self.classifier(t)               # B × num_classes
        return out
