import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import io
import numpy as np

class EmbeddingDataset(Dataset):
	def __init__(self, csv_path:str, data_path:str, transform = None):
		self.transform = transform 
		self.data_info = pd.read_csv(csv_path)
		self.data_len = len(self.data_info)
		self.data_path = data_path

	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		fetched_row = self.data_info.iloc[index]
		img = torch.tensor(np.load(self.data_path + "/" + fetched_row[1])[0])
		target = fetched_row[2]
		
		alternate_embeddings = fetched_row[3]
		alternate_images = {}
		for method_name, file_name in alternate_embeddings.items():
			alternate_images[method_name] = torch.tensor(np.load(self.data_path + "/" + file_name)[0])
		
		if self.transform:
			img = self.transform(img)
			for name, alt in alternate_images.copy():
				alternate_images[name] = self.transform(alt)
				
		return img, target, alternate_images


	def __len__(self):
		return self.data_len