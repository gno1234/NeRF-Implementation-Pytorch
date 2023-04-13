import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#########################################################
def min_max_norm(x: torch.Tensor ,min: int, max: int):
    
    x_min, x_max = x.min(), x.max()
    new_min, new_max = min, max
    x = (x - x_min)/(x_max - x_min)*(new_max - new_min) + new_min

    return(x)

def generate_rays(pose:torch.Tensor, image:torch.Tensor, focal_length:float):
    """
    args:
        pose (torch.Tensor) : torch.Size([1(data_batch),4,4]) Contains information about the direction and origin of the camera.
        image (torch.Tensor) : torch.Size([1(data_batch),height,width,3(rgb)]) Used to calculate the height and width of the image.
        focal_length (float) : calculate from 'camera angle'

    Returns:
        unit_directions (torch.Tensor) : torch.Size([image_batch,height,width,3(xyz)]) A unit vector representation of the direction of the ray for each pixel in the camera.
        origin (torch.Tensor) : torch..Size([3(xyz),]) The value of the camera's origin. 
    """
    pose = pose.squeeze() # Data loader loads data with batch size 1
    image = image.squeeze() # Data loader loads data with batch size 1

    h = image.size(0) #height
    w = image.size(1) #width

    # generate xy mesh grid pixel position (x,y) directions[x_index][y_index]
    x,y = torch.meshgrid(
        torch.arange(w),
        torch.arange(h),
        indexing = 'xy')
    
    x = min_max_norm(x,-w/2,w/2) # Adjust the x scale.
    y = min_max_norm(y,-h/2,h/2) # Adjust the y scale.
    z = torch.ones(w,h)*focal_length # The z-axis magnitude of the "direction vector" is equal to the focal length.

    directions = torch.stack((x,-y,-z),dim=-1) # Combine the x, y and z axes.

    vector_norm = torch.linalg.vector_norm(directions,dim=2,keepdim=True) # get magnitude of vector.

    unit_directions = directions/vector_norm #Makes the vector a unit vector.

    unit_directions = torch.einsum('ij,hwj->hwi', pose[:3,:3],  unit_directions) # transform the vector by camera pose matrix. #matrix-vector product
    
    origin = pose[:3,3] # Get the origin of the vector.

    return unit_directions, origin
#########################################################

#########################################################
def ray_sampling(near: float, far: float, num_samples: int, d, o):
    """
    args:
        near (float): distance of neareast sampling point
        far (float): distance of farthest sampling point
        num_samples (int) : number of sampling points
    Returns:
        sampling_point (torch.Tensor): cordinates of sampling points (each pixel)
        t (torch.Tensor): Sampling points on a one-dimensional 
    """

    t = torch.linspace(near, far, num_samples) # fixed interval sampling
    
    t = t.expand(d.size(0),d.size(1),num_samples) + torch.rand(d.size(0),d.size(1),num_samples)*(far-near)/(num_samples) # random position sampling

    sampling_points = torch.einsum('hwn,hwd->hwnd', t, d) # Expand the sampling points to the three-dimensional ray direction of each pixel.

    o = o.expand_as(sampling_points) 

    sampling_points = sampling_points + o # add camera origin

    d = d[...,None,:].expand_as(sampling_points) # Sampling points on a single ray have the same direction.

    return sampling_points, t, d
#########################################################


#########################################################
def cumsum_exclusive(x:torch.Tensor, dim:int):
    """
    x_i = 0 + x_1 + x_2 + x_(i-1)
    args:
        x (torch.Tensor):
        dim (int):
    Return:
        cumsum(torch.tensor) : size_as input
    """
    zeros =  torch.zeros_like(x).narrow(dim,0,1)
    x= x.narrow(dim,0,x.size(dim)-1)
    x= torch.cat((zeros,x),dim=dim)
    cumsum = torch.cumsum(x,dim=dim)
    return cumsum

def render_color(predictions:torch.Tensor, t:torch.Tensor, batch_size:int = 4096): 
    """
    render rgb image and depth map from output of nerf model.
    args:
        pridictions(torch.Tensor) : torch.size([batch_size,num_samples,4(rgb_sigma)])
        t(torch.Tensor) : torch.size([batch_size,num_samples)
    returns:
        C_r(torch.Tensor) : torch.size([batch_size,3(rgb)])
        depth_map(torch.Tensor) : torch.size([batch_size,])

    """
    # Slice the predictions into rgb and sigma.
    c_i = torch.sigmoid(predictions[..., :-1])
    sigma = torch.nn.functional.relu(predictions[..., -1])

    # The calculations were made in code by discretizing the formulas in the paper.
    # Get the distance of adjacent intervals.
    delta = t[..., 1:] - t[..., :-1] # delta.size() = torch.size([batch_size,num_samples - 1])
    delta = torch.cat((delta, torch.tensor([1e10]).to(device).expand(batch_size,1)), dim=-1) # last delta = big value
    
    sigma_delta = torch.einsum('bi,bi->bi',sigma,delta)

    eps = torch.tensor(1e-10) 
    T_i  = torch.exp(-cumsum_exclusive(sigma_delta+eps, dim = -1))
    alpha = 1-torch.exp(-sigma_delta)
    weights = torch.einsum('bi,bi->bi',T_i,alpha)
    C_r= torch.sum(torch.einsum('bn,bni->bni',weights,c_i) ,dim=-2)
    depth_map = torch.sum(weights * t, dim=-1)
    
    return C_r, depth_map, weights
#########################################################

#########################################################
def Hierarchical_Volume_Sampling(weights:torch.Tensor, t:torch.Tensor, d:torch.Tensor, o:torch.Tensor, n_fine:int =128):
    """
    Get the fine sampling points using the information from the coarse sampling and color calculations.
    Extract fine sampling points with weight probability from coarse sampling points.
    args:
        weights(torch.Tensor): The weights are generated from the rendering of the coarse sampling point.
        t(torch.Tensor): T from coarse sampling points
        d(torch.Tensor): d from coarse sampling points
        o(torch.Tensor): origin of camera
        n_fine(int): number of fine sampling points
    returns:
        points_fine(torch.Tensor): fine sampling point
        t_fine(torch.Tensor): fine t
        d_fine(torch.Tensor): fine d
    """
    
    batch_size = weights.size(0)
    d =d[:,0,:]

    PDF = (weights[:,:-1]+1e-10)/torch.sum(weights[:,:-1]+1e-10,dim=-1,keepdim=True) #The last weight is the weight from the last point to infinity, so we exclude it. and Sum the probabilities to 1
    sample_indices = torch.multinomial(PDF, n_fine-1, replacement=True) # Extract the indices of the fine sampling point.

    delta = t[..., 1:] - t[..., :-1]

    t_fine = torch.gather(t,1,sample_indices) # Get t and delta depending on the sampled index.
    delta_fine = torch.gather(delta,1,sample_indices) # Get t and delta depending on the sampled index.
    t_fine = t_fine + torch.rand_like(t_fine)*delta_fine # adds randomness to t.
   

    t_fine, _ = torch.sort(t_fine, dim=-1) #The sample index is sorted because it was randomly sampled.
    t_fine = torch.cat((t_fine,t[:,-1].unsqueeze(-1)),dim=-1) # Add the last excluded item.

    #Add the direction and origin of the camera to the fine sampling point.
    points_fine = torch.einsum('bn,bi->bni',t_fine,d) + o.expand(batch_size,n_fine,3).to(device)
    d_fine = d.unsqueeze(1).expand_as(points_fine)

    return points_fine, t_fine, d_fine
#########################################################