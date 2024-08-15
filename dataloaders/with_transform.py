import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision 
from PIL import Image
import os 


class TorchDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256,256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


root_path = "/kaggle/input/gan-getting-started"
data_path_monet = f"{root_path}/monet_jpg"
data_path_photo = f"{root_path}/photo_jpg"
MonetDataset = TorchDataset(data_path=data_path_monet, transform=transforms)