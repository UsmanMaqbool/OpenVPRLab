import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import faiss

# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False, alpha=100.0):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                    self.centroids[D:D+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual * soft_assign[:,D:D+1,:].unsqueeze(2)
            vlad[:,D:D+1,:] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, config, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        
        # Get the actual dataset from VPRDataModule
        dataset = cluster_ds._get_train_dataset()
        
        # Debugging: Print the length of the dataset
        # print(f"Length of dataset: {len(dataset)}")
        
        # Create a SubsetRandomSampler
        indices = np.random.choice(len(dataset), images_num, replace=False)
        random_sampler = SubsetRandomSampler(indices)
        
        # Debugging: Print the sampled indices
        # print(f"Sampled indices: {indices}")
        
        # Create DataLoader
        random_dl = DataLoader(dataset=dataset, num_workers=config['datamodule']['num_workers'],
                               batch_size=config['datamodule']['batch_size'], sampler=random_sampler)
        
        # Debugging: Print the number of batches
        # print(f"Number of batches: {len(random_dl)}")
        
        with torch.no_grad():
            backbone = backbone.eval()
            logging.debug("Extracting features to initialize NetVLAD layer")
            descriptors = np.zeros(shape=(descriptors_num, self.dim), dtype=np.float32)
            
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100, desc="Initializing NetVLAD")):
                # Debugging: Print the current iteration and batch size
                # print(f"Iteration: {iteration}, Batch size: {inputs.size(0)}")
                
                # inputs = inputs.to(config['device'])
                 # Reshape inputs from [40, 4, 3, 320, 320] to [40, 3, 320, 320]
                for channel in range(inputs.size(1)):  # Loop over all channels
                    channel_inputs = inputs[:, channel, :, :, :]  # Taking the current channel
                    outputs = backbone(channel_inputs)
                    norm_outputs = F.normalize(outputs, p=2, dim=1)
                    image_descriptors = norm_outputs.view(norm_outputs.shape[0], self.dim, -1).permute(0, 2, 1)
                    image_descriptors = image_descriptors.cpu().numpy()
                    batchix = iteration * config['datamodule']['batch_size'] * descs_num_per_image
                    for ix in range(image_descriptors.shape[0]):
                        sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                        startix = batchix + ix * descs_num_per_image
                        descriptors[startix:startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(self.dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        logging.debug(f"NetVLAD centroids shape: {kmeans.centroids.shape}")
        self.init_params(kmeans.centroids, descriptors)
        # self = self.to(args.device)