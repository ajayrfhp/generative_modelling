import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import image2tensor, tensor2image, display_image_side


class FaceDataset(Dataset):
    def __init__(self, input_category, output_category, folder):
        category_map = {'Angry': '0', 'Disgust': '1', 'Fear': '2', 'Happy': '3',
                        'Sad': '4', 'Surprise': '5', 'Neutral': '6'}
        root_path = f'../data/face_expression/{folder}/'
        self.input_root_path = root_path + category_map[input_category] + '/'
        self.output_root_path = root_path + category_map[output_category] + '/'
        self.input_files, self.output_files = os.listdir(
            self.input_root_path), os.listdir(self.output_root_path)
        self.length = min(len(self.input_files), len(self.output_files))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ])
        ])
        self.inverse_transform = lambda x: (x * 0.5 + 0.5) * 255

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        input_image = np.asarray(Image.open(
            self.input_root_path + self.input_files[i]))
        output_image = np.asarray(Image.open(
            self.output_root_path + self.output_files[i]))
        input_tensor = self.transform(image2tensor(input_image)).float()
        output_tensor = self.transform(image2tensor(output_image)).float()
        return input_tensor, output_tensor

    def display(self, i):
        input_tensor, output_tensor = self.__getitem__(i)
        input_tensor = self.inverse_transform(input_tensor)
        output_tensor = self.inverse_transform(output_tensor)
        input_image = tensor2image(input_tensor)
        output_image = tensor2image(output_tensor)

        display_image_side(input_image, output_image, show=True)
