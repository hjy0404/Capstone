#making train data set
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
data = ImageFolder('/content/drive/MyDrive/Colab Notebooks/cow/preprocessed_image', transform = transform)
train_dataset, val_dataset = random_split(data, [int(0.8*len(data)),len(data)-int(0.8*len(data))])
train_loader  = torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle = True, num_workers = 4, pin_memory = False)
validate_loader  = torch.utils.data.DataLoader(val_dataset,batch_size=4,shuffle = True, num_workers = 4,  pin_memory = False)
