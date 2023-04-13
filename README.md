# **NeRF-Implementation-Pytorch**

Pytorch implementation of NeRF based on the paper below.

Paper : ****NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis****

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

Tested in the Google colab environment

# Dataset

Implementation customized for the dataset below.

[nerf_synthetic - Google Drive](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)

The dataset_path be configured as shown below. The train folder and the transforms_train.json file are required.

```python
dataset_path
├── test
├── train
├── transforms_test.json
├── transforms_train.json
├── transforms_val.json
└── val
```

# Train

```python
!python nerf_train.py --dataset_path dataset/path 
```

Additionally, you can use the following arguments

```python
--iter_nupe (default = 25000)
--n_coarse (default = 64)
--n_fine (default = 128)
```

The train progress and model training results are automatically saved.

If the loss does not decrease in a few epochs, rerun the training.

Result

`100%|█████████▉| 24999/25000 [2:02:14<00:00,  3.59it/s]`
`loss :  0.004267157102003693`

![image.png](README%20md%2039e6beb13d8d480eb817b16ef875498f/image.png)

![loss and PSNR.png](README%20md%2039e6beb13d8d480eb817b16ef875498f/loss_and_PSNR.png)