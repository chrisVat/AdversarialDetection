import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import io

class AdversarialDataset(Dataset):
	def __init__(self, csv_path:str, data_path:str, transform = None):
		self.transform = transform 
		self.data_info = pd.read_csv(csv_path)
		self.data_len = len(self.data_info)
		self.data_path = data_path

		self.training = data_path.contains("train")
		self.cifar_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		
		self.cifar_trainset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=True, download=True, transform=self.cifar_transform)
		self.testset = torchvision.datasets.CIFAR10(root='./data/raw_data', train=False, download=True, transform=transform)
		if self.training:
			self.loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last=True)
		else:
			self.loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last=True)


	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		if index %2 == 1:
			return self.loader[index//2], 0
		else:
			fetched_row = self.data_info.iloc[index]
			img = torchvision.io.read_image(self.data_path + "/" + fetched_row[1]).float()
			target = fetched_row[2]
			if self.transform:
				img = self.transform(img)
			return img, target


	def __len__(self):
		return self.data_len