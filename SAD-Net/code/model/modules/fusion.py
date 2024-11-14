from torch import nn
import torch
from .cga import SpatialAttention, ChannelAttention, PixelAttention


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        if x.shape != y.shape:
            min_h = min(x.shape[2], y.shape[2])
            min_w = min(x.shape[3], y.shape[3])
            x = x[:, :, :min_h, :min_w]
            y = y[:, :, :min_h, :min_w]
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
    
if __name__ == '__main__':
    block = CGAFusion(128)  # 输入通道数，输出通道数
    x = torch.rand(1, 128, 1000, 1000)
    output = block(x)
    print('input :',x.size())
    print('output :', output.size())