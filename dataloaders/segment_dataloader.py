import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
from skimage.draw import polygon2mask

class CustomDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        with open(labels_file, 'r') as json_file:
            self.json_labels = [json.loads(line) for line in json_file]

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.json_labels)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, f"{self.json_labels[idx]['id']}.tif")
        image = Image.open(image_path)

        # Initialize mask
        mask = np.zeros((512, 512), dtype=np.float32)

        # Process annotations
        for annot in self.json_labels[idx]['annotations']:
            cords = annot['coordinates']
            if annot['type'] == "blood_vessel":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask[rr, cc] = 1

        # Convert PIL Image and mask to PyTorch tensor
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)  # Shape: [C, H, W]
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask