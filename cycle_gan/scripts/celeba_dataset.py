import os
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import image2tensor, tensor2image, display_image_side

class CelebADataset(Dataset):
    def __init__(self, data_root, label = 'Smiling'):
        self.image_folder = f'{data_root}img_align_celeba/img_align_celeba/'
        self.labels = pd.read_csv(f'{data_root}list_attr_celeba.csv')
        self.positive = self.labels[self.labels[label] == 1]
        self.negative = self.labels[self.labels[label] == -1]
        self.input_files = self.positive['image_id'].tolist()
        self.output_files = self.negative['image_id'].tolist()
        self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5,], [0.5, 0.5, 0.5,])
                        ])
        self.inverse_transform = lambda x : (x * 0.5 + 0.5) * 255

        
    def __getitem__(self, i):
        input_image = np.asarray(Image.open(self.image_folder + self.input_files[i]))
        output_image = np.asarray(Image.open(self.image_folder + self.output_files[i]))
        input_tensor = self.transform(image2tensor(input_image))
        output_tensor = self.transform(image2tensor(output_image))

        return input_tensor, output_tensor
    
    def __len__(self):
        return len(self.image_files)
    
    def display(self, i):
        input_tensor, output_tensor = self.__getitem__(i)
        input_image = tensor2image(self.inverse_transform(input_tensor)).astype(np.uint8)
        output_image = tensor2image(self.inverse_transform(output_tensor)).astype(np.uint8)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(input_image)
        ax[1].imshow(output_image)
        plt.show()
        

dataset = CelebADataset('../data/celebA/')
for i in range(2):
    dataset.display(i)
