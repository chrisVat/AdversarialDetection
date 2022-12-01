import pandas as pd
from PIL import Image

from torchvision import transforms
from torch.utils.data.dataset import Dataset

class AdversarialDataset(Dataset):
	def __init__(self, csv_path:str, data_path:str):
		self.to_tensor = transforms.ToTensor()
		self.data_info = pd.read_csv(csv_path)
		self.data_len = len(self.data_info)
		self.data_path = data_path

	def __getitem__(self, index):
		fetched_row = self.data_info.iloc[index]
		img_as_tensor = self.to_tensor(Image.open(self.data_path + "/" + fetched_row[1]))

		to_return = {
			'image': img_as_tensor,
			'actual_class': fetched_row[2],
			'is_attacked': fetched_row[3],
			'model_prediction': fetched_row[4],
		}
		return to_return


	def __len__(self):
		return self.data_len