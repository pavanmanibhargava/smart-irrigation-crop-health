import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CropHealthDataset(Dataset):

    def __init__(self, csv_file, transform=None, class_to_index=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        if class_to_index is None:
            classes = sorted(self.data["class"].unique())

            self.class_to_index = {
                class_name: index
                for index, class_name in enumerate(classes)
            }

        else:
            self.class_to_index = class_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path = self.data.iloc[index]["image_path"]
        class_name = self.data.iloc[index]["class"]

        image = Image.open(image_path).convert("RGB")

        label = self.class_to_index[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
