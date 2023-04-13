import torch
import numpy as np

from nerf_processing import generate_rays, ray_sampling, render_color, Hierarchical_Volume_Sampling

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


############################
def one_iter(pose, image, f, n_coarse=64, n_fine=128, batch_size = 8192):
    """
    During training, pixels from an image are randomly selected by the batch size.
    """
    d_cam, o_cam = generate_rays(pose, image, f)
    points_eval, t_eval, d_eval = ray_sampling(2, 7, n_coarse, d_cam, o_cam)

    image = image.squeeze().reshape(-1,3)
    points_eval = points_eval.reshape(-1,n_coarse,3)
    t_eval = t_eval.reshape(-1,n_coarse)
    d_eval = d_eval.reshape(-1,n_coarse,3)

    #random indices
    indices = torch.randint(low=0, high = image.size(0), size=(batch_size,))

    image = torch.index_select(image, dim = 0, index = indices)
    points_coarse = torch.index_select(points_eval, dim = 0, index = indices)
    t_coarse = torch.index_select(t_eval, dim = 0, index = indices)
    d_coarse = torch.index_select(d_eval, dim = 0, index = indices)

    image = image.to(device)
    points_coarse = points_coarse.to(device)
    t_coarse = t_coarse.to(device)
    d_coarse = d_coarse.to(device)
     
    out = model(points_coarse,d_coarse)
    rgb, depth_map, weights_coarse = render_color(out, t_coarse, batch_size = batch_size)
    loss_coarse = torch.nn.functional.mse_loss(rgb, image)

    
    points_fine, t_fine, d = Hierarchical_Volume_Sampling(weights_coarse,t_coarse,d_coarse,o_cam, n_fine=n_fine)
    out = model(points_fine,d)
    rgb, depth_map, weights = render_color(out, t_fine, batch_size = batch_size)
    loss_fine = torch.nn.functional.mse_loss(rgb, image)
    
    loss = loss_coarse + loss_fine

    return rgb, depth_map, loss
############################



############################
def eval_image(pose, image, f,n_coarse=64, n_fine=128, batch_size=8000):
    """
    When eval, sequentially select pixels from one image by the batch size.
    """
    
    d_cam, o_cam = generate_rays(pose, image, f)
    points_eval, t_eval, d_eval = ray_sampling(2, 7, n_coarse, d_cam, o_cam)

    points_eval = points_eval.reshape(-1,n_coarse,3)
    t_eval = t_eval.reshape(-1,n_coarse)
    d_eval = d_eval.reshape(-1,n_coarse,3)
    
    image=image.squeeze()
    image_size = image.reshape(-1,3).size(0)
    n = int(image_size/batch_size)

    rgb_coarse_list = []
    rgb_fine_list =[]
    depth_list = []

    #weights_list = []

    for i in range(n):

        indices = torch.arange(batch_size*i,batch_size*(i+1))

        points_coarse = torch.index_select(points_eval, dim = 0, index = indices)
        t_coarse = torch.index_select(t_eval, dim = 0, index = indices)
        d_coarse = torch.index_select(d_eval, dim = 0, index = indices)

        points_coarse = points_coarse.to(device)
        t_coarse = t_coarse.to(device)
        d_coarse = d_coarse.to(device)

        ############## coarse sampling ##################        
        with torch.no_grad():
            out = model(points_coarse,d_coarse)
        rgb_coarse, depth_coarse, weights_coarse = render_color(out, t_coarse, batch_size = batch_size)
        #################################################

        ############### fine sampling ###################
        points_fine, t_fine, d_fine = Hierarchical_Volume_Sampling(weights_coarse,t_coarse,d_coarse,o_cam, n_fine=n_fine)

        with torch.no_grad():
            out = model(points_fine,d_fine)
        rgb_fine, depth_fine, weights_fine = render_color(out, t_fine, batch_size = batch_size)
        ################################################

        rgb_coarse_list.append(rgb_coarse.cpu())
        rgb_fine_list.append(rgb_fine.cpu())
        depth_list.append(depth_fine.cpu())
        
        #weights_list.append(weights_coarse.cpu())

    ##### display image ######

    
    image_coarse_recon = torch.cat(rgb_coarse_list,dim=0).reshape_as(image).numpy()
    image_fine_recon = torch.cat(rgb_fine_list,dim=0).reshape_as(image).numpy()
    depth_map = torch.cat(depth_list,dim=0).reshape(image.size(0),image.size(1)).numpy()

    #weights_map = torch.cat(weights_list,dim=0).reshape(image.size(0),image.size(1),n_coarse).numpy()
    #weights_map = weights_map.mean(-1)
    

    image = image.cpu().numpy()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))

    # ground truth image
    ax1.imshow(image, origin='upper')
    ax1.set_title('image_gt', fontsize=10)

    # reconstruction from coarse sampling
    ax2.imshow(image_coarse_recon, origin='upper')
    ax2.set_title('image_coarse_recon', fontsize=10)

    # reconstruction from fine sampling
    ax3.imshow(image_fine_recon,origin='upper')
    ax3.set_title('image_fine_recon', fontsize=10)
    
    # depth_map
    ax4.imshow(depth_map,cmap='gray', origin='upper',vmin = 0, vmax=8)
    ax4.set_title('depth_map', fontsize=10)

    #ax5.imshow(weights_map, cmap='gray', origin='upper')
    #ax5.set_title('weights_mean_map', fontsize=10)

    fig.tight_layout()
    
    plt.show()

    return
############################

def mse_to_psnr(mse):
    return 20 * np.log10(1.0 / np.sqrt(mse))