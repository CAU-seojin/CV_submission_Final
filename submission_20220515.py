import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding,
                 dilation=(1,1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride, padding,
                              dilation, groups, bias)
        if bn_acti:
            self.bn_prelu = nn.Sequential(
                nn.BatchNorm2d(nOut, eps=1e-3),
                nn.SELU(inplace=True)
            )
        self.bn_acti = bn_acti
    def forward(self, x):
        x = self.conv(x)
        if self.bn_acti:
            x = self.bn_prelu(x)
        return x

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn   = nn.BatchNorm2d(nIn, eps=1e-3)
        self.act  = nn.SELU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(x))

def Split(x, p):
    c = x.size(1)
    c1 = round(c * (1-p))
    return x[:, :c1], x[:, c1:]

class TCA(nn.Module):
    def __init__(self, c, d=1, kSize=3, dkSize=3):
        super().__init__()
        self.conv3x3   = Conv(c,   c,   kSize, 1, padding=1, bn_acti=True)
        self.dconv3x3  = Conv(c,   c, (dkSize,dkSize), 1,
                              padding=(1,1), groups=c, bn_acti=True)
        self.ddconv3x3 = Conv(c,   c, (dkSize,dkSize), 1,
                              padding=(d,d), groups=c,
                              dilation=(d,d), bn_acti=True)
        self.bp = BNPReLU(c)
    def forward(self, x):
        b  = self.conv3x3(x)
        b1 = self.dconv3x3(b)
        b2 = self.ddconv3x3(b)
        return self.bp(b + b1 + b2)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PCT_residual_SE(nn.Module):
    def __init__(self, nIn, d=1, p=0.5, se_red=16):
        super().__init__()
        self.p = p
        c      = nIn - round(nIn*(1-p))
        self.tca      = TCA(c, d)
        self.conv1x1  = Conv(nIn, nIn, 1,1,0, bn_acti=True)
        self.res_conv = nn.Conv2d(nIn, nIn, 1, bias=False)
        self.res_bn   = nn.BatchNorm2d(nIn)
        self.se       = SELayer(nIn, reduction=se_red)
    def forward(self, x):
        x1, x2 = Split(x, self.p)
        y2     = self.tca(x2)
        y      = self.conv1x1(torch.cat([x1, y2], dim=1))
        res    = self.res_bn(self.res_conv(x))
        out    = F.relu(y + res, inplace=True)
        return self.se(out)

class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        nConv = nOut-nIn if nIn< nOut else nOut
        self.conv3x3  = Conv(nIn, nConv, 3,2,1)
        self.pool     = nn.MaxPool2d(2,2,0)
        self.bn_prelu = BNPReLU(nOut)
        self.nIn, self.nOut = nIn, nOut
    def forward(self, x):
        out = self.conv3x3(x)
        if self.nIn < self.nOut:
            out = torch.cat([out, self.pool(x)], dim=1)
        return self.bn_prelu(out)

class DWSepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, d=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, k,1,p,d,groups=in_ch,bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch,1,bias=False)
        self.bn  = nn.BatchNorm2d(out_ch, momentum=0.1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x)
        return self.act(self.bn(x))

class TinyASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.br1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,1,bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.br2 = DWSepConv(in_ch, out_ch, k=3, p=6, d=6)
        self.proj= nn.Sequential(
            nn.Conv2d(out_ch*2, out_ch,1,bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        y1 = self.br1(x)
        y2 = self.br2(x)
        return self.proj(torch.cat([y1,y2],1))

class DAD(nn.Module):
    def __init__(self, c2, c1, num_classes):
        super().__init__()
        self.conv1x1_c   = Conv(c2,   c1, 1,1,0, bn_acti=True)
        self.conv1x1_neg = Conv(c1,   c1, 1,1,0, bn_acti=True)
        self.conv3x3     = Conv(c1,   c1, 3,1,1, groups=c1, bn_acti=True)
        self.classifier  = nn.Conv2d(c1, num_classes, 1)

    def forward(self, X, L):
        X_map = torch.sigmoid(X)
        Yc    = self.conv1x1_c(L)
        Yc_map= torch.sigmoid(Yc)
        Neg   = self.conv1x1_neg(-Yc_map)
        F_rg  = Neg*Yc_map + Yc
        F_rg  = F.interpolate(F_rg, size=X_map.shape[2:],
                              mode='bilinear', align_corners=False)
        out   = self.conv3x3(X_map * F_rg)
        return self.classifier(out)

class submission_20220515(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        block_1 = 3
        block_2 = 7
        C = 32
        P = 0.5
        self.C, self.P = C, P

        # Init + LCNet Stage1 (→1/4)
        self.Init = nn.Sequential(
            Conv(in_channels, C, 3, 2, 1, bn_acti=True),
            Conv(C, C, 3, 1, 1, bn_acti=True),
            Conv(C, C, 3, 1, 1, bn_acti=True),
        )
        self.Stage1 = nn.Sequential(
            DownSamplingBlock(C, 2*C),
            *[PCT_residual_SE(2*C, d=2, p=P) for _ in range(block_1)]
        )
        # LCNet Stage2 (→1/8)
        self.Stage2 = nn.Sequential(
            DownSamplingBlock(2*C, 4*C),
            *[PCT_residual_SE(4*C, d=4, p=P) for _ in range(block_2)]
        )

        # TinyASPP
        self.aspp = TinyASPP(in_ch=4*C, out_ch=2*C)
        # low-level skip proj
        self.low_proj = nn.Sequential(
            nn.Conv2d(2*C, 2*C, 1, bias=False),
            nn.BatchNorm2d(2*C),
            nn.ReLU(inplace=True)
        )
        # attention decoder
        self.dad = DAD(c2=2*C, c1=2*C, num_classes=num_classes)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        f0 = self.Init(x)       # 1/2
        f1 = self.Stage1(f0)    # 1/4
        f2 = self.Stage2(f1)    # 1/8

        a = self.aspp(f2)
        a = F.interpolate(a, size=f1.shape[2:],
                           mode='bilinear', align_corners=False)
        l = self.low_proj(f1)
        out = self.dad(a, l)    # 1/4 → logits

        return F.interpolate(out, size=(H, W),
                             mode='bilinear', align_corners=False)