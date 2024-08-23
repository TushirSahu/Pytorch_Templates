import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class PaintingsDataset(Dataset):
    '''
    Basic dataset class, will be used to manipulate the monet dataset 
    
    __init__ args:          folder_path       :       path to folder with dataset images
                            transform         :       transform module from torchvision for data augmentation
                            __len__           :       number of images in dataset
                            __get_item__      :       returns image at index idx transformed
                            __get_crude_item_ :       returns image at index idx 
                            
                
    '''
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
    def __get_crude_item__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        return image