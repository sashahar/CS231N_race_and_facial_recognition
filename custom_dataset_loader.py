from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os

#Custom Pytorch dataloader used to keep track of gender, race, age and image name for each image tensor
#Use the tutorials and examples at the following Github link for guidance: 
#https://github.com/utkuozbulak/pytorch-custom-dataset-examples

class gender_race_dataset(Dataset):
    def __init__(self, data_path, path_to_img, transf = None):
        
        if(transforms == None):
            self.trans = transforms.Compose([transforms.ToTensor()])
        else:
            self.trans = transf
        
        self.raw_data = pd.read_csv(os.path.join(os.getcwd(),data_path))
        self.age_labels = np.asarray(self.raw_data.iloc[:,0])
        self.gender_labels = np.asarray(self.raw_data.iloc[:,1])
        self.race_labels = np.asarray(self.raw_data.iloc[:,2])
        self.img_names = np.asarray(self.raw_data.iloc[:,3])
        self.num_samples = len(self.raw_data.index)
        self.path_to_img = path_to_img
        self.gender_list = ["male", "female"]

         
    def __getitem__(self, index):
        gender = self.gender_labels[index]
        race = self.race_labels[index]
        age = self.age_labels[index]
        img_name = self.img_names[index]
        full_img = "_".join([str(age), str(gender), str(race), str(img_name)])
        
        abs_path = os.path.join(os.getcwd(), os.path.join(self.path_to_img, os.path.join(self.gender_list[int(gender)],full_img)))
        
        img = Image.open(abs_path)
        transformed_img = self.trans(img)
        
        if race != 0:
            binary_race = 1
        else:
            binary_race = 0
        
        return (int(gender), int(binary_race), full_img, transformed_img)

    def __len__(self):
        return self.num_samples  