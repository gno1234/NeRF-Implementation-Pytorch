import torch
from torch.utils.data import DataLoader
import numpy as np

from nerf_model import Nerf_network
from nerf_dataset import CustomDataset
from nerf_train_funtion import eval_image, one_iter, mse_to_psnr

import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type = str)
parser.add_argument("--iter_num", type = int, default = 25000)
parser.add_argument("--n_coarse", type = int,default = 64)
parser.add_argument("--n_fine", type = int,default = 128)
args = parser.parse_args()

print(os.getcwd())
os.makedirs('./progress/', exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

model = Nerf_network()
model.to(device)
model.apply(model._init_weights)

train_dataset = CustomDataset(path = args.dataset_path ,image_size =200)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

pose, image = next(iter(train_dataloader))
image_height = image.size(1)

angle = train_dataset.camera_angle
angle_deg = angle * 180/np.pi
f = (image_height*0.5)/np.tan(angle*0.5)


#model = torch.compile(model)

lr = 5e-4
iter_num = args.iter_num
n_coarse = args.n_coarse
n_fine = args.n_fine


model.eval()
pose, image = next(iter(train_dataloader))
fig_eval = eval_image(model, pose, image, f, n_coarse=n_coarse, n_fine = n_fine, batch_size=4000)
print(os.getcwd())
plt.savefig('./progress/initial_state.png')

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

loss_list = [] # for calculating average loss per epoch
loss_record = [] # for record avg loss

for i in tqdm(range(iter_num)):

    pose, image = next(iter(train_dataloader))
    rgb, depth_map, loss = one_iter(model, pose, image, f, n_coarse=n_coarse, n_fine=n_fine, batch_size=4096)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_list.append(loss.item())

    if (i+1) % len(train_dataloader) == 0:
        
        avg_loss = sum(loss_list)/100
        print(' loss : ', avg_loss)
        loss_record.append(avg_loss)
        loss_list=[]

        scheduler.step(avg_loss)

        print('eval...')
        model.eval()
        pose, image = next(iter(train_dataloader))
        with torch.no_grad():
            fig_eval = eval_image(model, pose, image, f, n_coarse=n_coarse, n_fine=n_fine, batch_size=4000)
            path = "./progress/"+str(i+1)+'iter_eval_image.png'
            plt.savefig(path)
        model.train()

        
        # plot loss and PSNR ######
        psnr_record = list(map(mse_to_psnr, loss_record))

        x = range(len(loss_record))

        fig, ax1 = plt.subplots(figsize=(4, 2))
        ax1.plot(x, loss_record, 'b-')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(x, psnr_record, 'r-')
        ax2.set_ylabel('PSNR', color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        path = "./progress/"+str(i+1)+'iter_eval_loss_PSNR.png'
        plt.savefig(path)
        ##########################

os.makedirs('./result/', exist_ok=True)
torch.save(model.state_dict(), "./result/nerf_train_result.pt")