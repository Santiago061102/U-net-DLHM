from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch import nn
from torch.nn import functional as F
from torch.fft import fft2, fftshift, rfft2
import torch, kornia

class MSELoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mse=nn.MSELoss()

    def forward(self, fake, real):
        loss = self.mse(fake, real)
        return loss


# class LapLoss(nn.Module):
#     def __init__(self,):
#         super().__init__()
#         self.mse=nn.MSELoss()
    
#     def forward(self, pred, real):
#         lap_tar = kornia.filters.laplacian(pred, kernel_size=3)
#         lap_real = kornia.filters.laplacian(real, kernel_size=3)
        
#         loss = self.mse(lap_real, lap_tar)
#         return loss


class SSIMLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.ssim=StructuralSimilarityIndexMeasure()

    def forward(self, fake, real):
        loss = 1 - self.ssim(fake, real)
        return loss


class FourierLossMSE(nn.Module):
    def __init__(self, lmb1 = 0.5, lmb2 = 0.5):
        super().__init__()
        self.mse=nn.MSELoss()
        self.lmb1 = lmb1
        self.lmb2 = lmb2


    def forward(self, fake, real):
        fft_real = torch.real(fft2(real))
        #fft_real = (fft_real - torch.min(fft_real))/(torch.max(fft_real) - torch.min(fft_real))

        fft_fake = torch.real(fft2(fake))
        #fft_fake = (fft_fake - torch.min(fft_fake))/(torch.max(fft_fake) - torch.min(fft_fake))

        loss = self.mse(real, fake)*self.lmb1  + self.mse(fft_real, fft_fake)*self.lmb2
        return loss

class FourierLossSSIM(nn.Module):
    def __init__(self,):
        super().__init__()
        self.ssim=StructuralSimilarityIndexMeasure()

    def forward(self, fake, real):
        fft_real = torch.abs(fft2(real))
        fft_fake = torch.abs(fft2(fake))
        loss = (1 - self.ssim(real, fake))*0.5 + (1 - self.ssim(fft_real, fft_fake))*0.5
        return loss



class HybLoss(nn.Module):
    def __init__(self, lmb1 = 0.5, lmb2 = 0.5):
        super().__init__()
        self.mse=nn.MSELoss()
        self.ssim=StructuralSimilarityIndexMeasure()
        self.lmb1 = lmb1
        self.lmb2 = lmb2

    def forward(self, fake, real):

        mseLoss = self.mse(real, fake)
        ssimLoss = 1- self.ssim(real, fake)

        lap_tar = kornia.filters.laplacian(fake, kernel_size=3)
        lap_real = kornia.filters.laplacian(real, kernel_size=3)
        mse_lap = self.mse(lap_real, lap_tar)

        return mseLoss*self.lmb1 + ssimLoss*self.lmb2




###########################################################################################




class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes,  kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)

        self.bn=None
        if norm:
            self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)

        if self.bn is not None:
            fx = self.bn(fx)

        return fx

class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, inplanes, outplanes,  kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)

        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)

    def forward(self, x):
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)

        return fx

class UNet(nn.Module):
    """Unet-like Encoder-Decoder model"""
    def __init__(self, filters):
        super().__init__()

        

        self.encoder1 = nn.Conv2d(1, filters,  kernel_size=4, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(filters)
        self.encoder2 = EncoderBlock(filters, filters*2, norm=True)
        self.encoder3 = EncoderBlock(filters*2, filters*4, norm=True)
        self.encoder4 = EncoderBlock(filters*4, filters*8, norm=True)
        self.encoder5 = EncoderBlock(filters*8, filters*8, norm=True)
        self.encoder6 = EncoderBlock(filters*8, filters*8, norm=True)
        self.encoder7 = EncoderBlock(filters*8, filters*8, norm=True)
        self.encoder8 = EncoderBlock(filters*8, filters*8, norm=True)

        self.decoder8 = DecoderBlock(filters*8, filters*8, dropout=True)
        self.decoder7 = DecoderBlock(2*filters*8, filters*8, dropout=True)
        self.decoder6 = DecoderBlock(2*filters*8, filters*8, dropout=True) ## Here was the dropout original
        self.decoder5 = DecoderBlock(2*filters*8, filters*8, dropout=True)
        self.decoder4 = DecoderBlock(2*filters*8, filters*4, dropout=True)
        self.decoder3 = DecoderBlock(2*filters*4, filters*2, dropout=True)
        self.decoder2 = DecoderBlock(2*filters*2, filters, dropout=True)
        self.decoder1 = nn.ConvTranspose2d(2*filters, 1,  kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # encoder forward
        e1 = self.encoder1(x)
        e1 = self.batch_norm(e1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        # decoder forward + skip connections
        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)

        return torch.tanh(d1)
