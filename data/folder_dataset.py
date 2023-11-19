import random
import torch

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

import torchvision.transforms as transforms


# Define the rotation function
def rotate_image(image):
    angles = [0, 90, 180, 270]
    angle = angles[random.randint(0, 3)]  # Randomly select one of the angles
    return image.rotate(angle)


def random_crop(image, size=256):
    x = random.randint(0, image.size[0] - size) if size < image.size[0] else 0
    y = random.randint(0, image.size[1] - size) if size < image.size[1] else 0

    return image.crop((x, y, x + size, y + size))



class ImageNetDataset(Dataset):
    def __init__(self, stage, opts):
        super().__init__()
        self.imagenet_dir = opts.dataset_path
        self.transform = transforms.Compose([
            transforms.Lambda(random_crop),
            transforms.Resize((256, 256)),  # Rescale the larger dimension to 256 while maintaining the aspect ratio
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
            transforms.Lambda(rotate_image),
            transforms.ToTensor(),  # Convert the image to a tensor
        ])
        self.dataset = ImageFolder(self.imagenet_dir, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cameras = [{'P': torch.eye(4), 'Pinv': torch.eye(4), 'K': torch.eye(4), 'Kinv': torch.eye(4)}]

        images, _ = self.dataset[idx]

        return {"idx": idx, "path": f'{idx}', "img_id": idx, "images": [images], "cameras": cameras}
    
    def totrain(self, epoch=0):
        pass

    def toval(self, epoch=0):
        pass

    def totest(self, epoch=0):
        pass
