from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch import nn
from torch.nn import functional as F
from torch.fft import fft2, fftshift, rfft2
import torch, kornia
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity



class Self_Attn(nn.Module):  ### Attention mechanism
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        #print(x.dtype, self.query_conv.weight.dtype, self.gamma.dtype)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        

        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=False):
        super().__init__()
        padding = kernel_size // 2

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        ]

        if dropout:
            layers.append(nn.Dropout2d(p=0.25, inplace=False))

        self.conv_op = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_op(x)
    

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p
    

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, attnM = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dropout= True)
        self.attn = Self_Attn(in_channels, 'relu')
        self.attnM = attnM


    def forward(self, x1, x2):
        x1 = self.up(x1)        
        x = torch.cat([x1, x2], 1)
        #if self.attnM:
        #x, _ = self.attn(x)
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, filters, kernel_size=3, attn = False):
        super().__init__()
        self.attn = attn
        self.down_convolution1 = DownSample(1, filters, kernel_size)
        self.down_convolution2 = DownSample(filters, filters*2, kernel_size)
        self.down_convolution3 = DownSample(filters*2, filters*4, kernel_size)
        self.down_convolution4 = DownSample(filters*4, filters*8, kernel_size)

        self.bottle_neck = DoubleConv(filters*8, filters*16, kernel_size, dropout=True)

        self.up_convolution1 = UpSample(filters*16, filters*8, kernel_size, attnM = False)
        self.up_convolution2 = UpSample(filters*8, filters*4, kernel_size, attnM = False)
        self.up_convolution3 = UpSample(filters*4, filters*2, kernel_size, attnM = False)
        self.up_convolution4 = UpSample(filters*2, filters, kernel_size, attnM = False)

        self.attn1 = Self_Attn(filters*16, 'relu')
        # self.attn2 = Self_Attn(filters*8, 'relu')

        # Use same kernel size for final output layer
        padding = kernel_size // 2
        self.out = nn.Sequential(
            nn.Conv2d(filters, filters//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(filters//2, 1, kernel_size=1)  # 1x1 para la salida final
        )
        #self.out = nn.Conv2d(filters, out_channels=1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        down1, p1 = self.down_convolution1(x)
        down2, p2 = self.down_convolution2(p1)
        down3, p3 = self.down_convolution3(p2)
        down4, p4 = self.down_convolution4(p3)

        b = self.bottle_neck(p4)
        if self.attn:
            b, attn = self.attn1(b)
        #print(b.shape)

        up1 = self.up_convolution1(b, down4)
        up2 = self.up_convolution2(up1, down3)
        up3 = self.up_convolution3(up2, down2)
        up4 = self.up_convolution4(up3, down1)

        out = self.out(up4)
        return torch.tanh(out)

