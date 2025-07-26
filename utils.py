from torch.utils.data.dataset import Dataset
import torch, os, glob, json
from datetime import datetime
from PIL import Image
import numpy as np
from torch import nn

class Data(Dataset):
    def __init__(self,
                 root: str="/home/spascuasm/models",
                 transform=None,
                 num_images=5000,
                 mode: str='train'):
        self.root=root
        self.mode = mode
        if self.mode == 'train':
            self.files_src=sorted(glob.glob(f"{root}/src_ph/*.png"))[0:int(num_images*0.8)]
            self.files_tar=sorted(glob.glob(f"{root}/tar_ph/*.png"))[0:int(num_images*0.8)]
        else:
            self.files_src=sorted(glob.glob(f"{root}/src_ph/*.png"))[int(num_images*0.8):num_images]
            self.files_tar=sorted(glob.glob(f"{root}/tar_ph/*.png"))[int(num_images*0.8):num_images]
        self.transform=transform

    def __len__(self,):
        return len(self.files_src)

    def __getitem__(self, idx):
        imgA = Image.open(self.files_src[idx]).convert('L')
        imgB = Image.open(self.files_tar[idx]).convert('L')

        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
            
        return imgA, imgB


class Logger():
    def __init__(self,
                 exp_name: str='/home/spascuasm/models',
                 filename: str=None):
        self.exp_name=exp_name
        self.cache={}
        if not os.path.exists(exp_name):
            os.makedirs(exp_name, exist_ok=True)
        self.date=datetime.today().strftime("%B_%d_%Y_%I_%M%p")
        if filename is None:
            self.filename=self.date
        else:
            self.filename="_".join([self.date, filename])
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)

    def add_scalar(self, key: str, value: float, t: int):
        if key in self.cache:
            self.cache[key][t] = value
        else:
            self.cache[key] = {t:value}
        self.update()
        return None

    def historical(self, info: dict):
      self.cache[self.filename] = info
      self.update()
      return None

    def save_weights(self, state_dict, model_name: str='model'):
        fpath = f"{self.exp_name}/{self.filename}_{model_name}.pt"
        torch.save(state_dict, fpath)
        return None

    def update(self,):
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        return None

    def close(self,):
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        self.cache={}
        return None

class Transformer(object):
    """"Transform"""
    def __init__(self,):
        pass
    def __call__(self, imgA, imgB=None):
        pass

class Compose(Transformer):
    """Compose transforms"""
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms=transforms

    def __call__(self, imgA, imgB=None):
        if imgB is None:
            for transform in self.transforms:
                imgA = transform(imgA, imgB)
            return imgA
        for transform in self.transforms:
            imgA, imgB = transform(imgA, imgB)
        return imgA, imgB

class Resize(Transformer):
    """Resize imageA and imageB"""
    def __init__(self, size=(256, 256)):
        """
        :param: size (default: tuple=(256, 256)) - target size
        """
        super().__init__()
        self.size=size

    def __call__(self, imgA, imgB=None):
        imgA = imgA.resize(self.size)
        if imgB is None:
            return imgA
        imgB = imgB.resize(self.size)
        return imgA, imgB

class ToTensor(Transformer):
    """Convert imageA and imageB to torch.tensor"""
    def __init__(self,):
        super().__init__()

    def __call__(self, imgA, imgB=None):
        imgA = np.array(imgA)/255.
        imgA = torch.from_numpy(imgA).unsqueeze(0).float().permute(0, 2, 1)
        if imgB is None:
            return imgA
        imgB = np.array(imgB)/255.
        imgB = torch.from_numpy(imgB).unsqueeze(0).float().permute(0, 2, 1)
        return imgA, imgB

def initialize_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif isinstance(layer, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)
    return None
