import torch
from torch.utils.data import Dataset

import numpy as np
import json
import os
import cv2

class CustomDataset(Dataset):
    def __init__(self, path, image_size = 400):
    
        self.path = path
        self.path_json = os.path.join(path,"transforms_train.json")

        with open(self.path_json, 'r') as file:
            train_json = json.load(file)

        self.frames_length = len(train_json['frames'])

        os.chdir(self.path)

        self.camera_angle = train_json['camera_angle_x']

        self.poses = [] 
        for i in range(self.frames_length):
            pose = np.array(train_json['frames'][i]['transform_matrix'])
            self.poses.append(pose)

        self.images = []
        for i in range(self.frames_length):
            image_path = train_json['frames'][i]['file_path']
            img = cv2.imread(image_path+".png")
            img = cv2.resize(img, (image_size, image_size))
            img_np = np.array(img)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_np = img_np/255
            self.images.append(img_np)

    def __len__(self):
        return self.frames_length

    def __getitem__(self, idx):
        pose = torch.Tensor(self.poses[idx])
        image = torch.Tensor(self.images[idx])
        return pose, image
    
    def camera_angle(self):
        return self.camera_angle