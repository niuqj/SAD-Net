from torch import nn
import torch
from model.modules.sdeconv import SDEConv
from model.modules.fga import FGA


class SDEBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size=3):
        super(SDEBlock, self).__init__()
        self.conv1 = SDEConv(dim)
        #self.conv1 = nn.Conv2d(dim,dim,kernel_size,padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim,dim,kernel_size,bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = res + x
        return res


    
class SDEABlock(nn.Module):
    def __init__(self, conv, dim, kernel_size=3):
        super(SDEABlock, self).__init__()
        #self.conv1 = nn.Conv2d(dim,dim,kernel_size,padding=1)
        self.conv1 = SDEConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.a = FGA(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res+x
        att = self.a(res)
        cat  = att+x
        res = self.conv2(cat)
        return res

