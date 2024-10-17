import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from tqdm import tqdm
import faiss
from torch.nn import init
from torchvision import transforms
from torchvision.ops import masks_to_boxes
class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """Aggregate node neighbors
        Args:
            input_dim: the dimension of the input feature
            output_dim: the dimension of the output feature
            use_bias: whether to use bias (default: {False})
            aggr_method: neighbor aggregation method (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = torch.amax(neighbor_feature, 1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias
        return neighbor_hidden
    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)
class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.gelu,
                 aggr_neighbor_method="sum",
                 aggr_hidden_method="concat"):
        """SageGCN layer definition
        Args:
            input_dim: the dimension of the input feature
            hidden_dim: dimension of hidden layer features,
                When aggr_hidden_method=sum, the output dimension is hidden_dim
                When aggr_hidden_method=concat, the output dimension is hidden_dim*2
            activation: activation function
            aggr_neighbor_method: neighbor feature aggregation method, ["mean", "sum", "max"]
            aggr_hidden_method: update method of node features, ["sum", "concat"]
        """
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden
class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_neighbors_list):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.num_neighbors_list = num_neighbors_list 
        self.num_layers = len(num_neighbors_list)  
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0])) 
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index+1])) 
        self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
    def forward(self, node_features_list):
        hidden = node_features_list
        subfeat_size = int(hidden[0].shape[1]/self.input_dim) 
        gcndim = int(self.input_dim) 
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_node_features = hidden[hop] 
                src_node_num = len(src_node_features) 
                neighbor_node_features = hidden[hop + 1] \
                    .view((src_node_num, self.num_neighbors_list[hop], -1))
                for j in range(subfeat_size):    
                    h_x = gcn(src_node_features[:,j*gcndim:j*gcndim+gcndim], neighbor_node_features[:,:,j*gcndim:j*gcndim+gcndim])
                    if (j==0):
                        h = h_x; 
                    else:
                        h = torch.concat([h, h_x],1) 
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]
    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
        
class applyGNN(nn.Module):
    def __init__(self):
        super(applyGNN, self).__init__()
        self.input_dim = 4096 
        self.hidden_dim = [2048,2048]
        self.num_neighbors_list = [5]
        self.graph = GraphSage(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                  num_neighbors_list=self.num_neighbors_list)
    def forward(self, x):
        gvlad = self.graph(x)
        return gvlad
class SelectRegions(nn.Module):
    def __init__(self, NB, Mask):
        super(SelectRegions, self).__init__()
        self.NB = NB
        self.mask = Mask
        self.visualize = False
        
    def relabel(self, img):
        """
        This function relabels the predicted labels so that cityscape dataset can process
        :param img: The image array to be relabeled
        :return: The relabeled image array
        """
        ### Road 0 + Sidewalk 1
        img[img == 1] = 1
        img[img == 0] = 1

        ### building 2 + wall 3 + fence 4
        img[img == 2] = 2
        img[img == 3] = 2
        img[img == 4] = 2
        

        ### vegetation 8 + Terrain 9
        img[img == 9] = 3
        img[img == 8] = 3

        ### Pole 5 + Traffic Light 6 + Traffic Signal
        img[img == 7] = 4
        img[img == 6] = 4
        img[img == 5] = 4
        
        ### Sky 10
        img[img == 10] = 5
        

        ## Rider 12 + motorcycle 17 + bicycle 18
        img[img == 18] = 255
        img[img == 17] = 255
        img[img == 12] = 255


        # cars 13 + truck 14 + bus 15 + train 16
        img[img == 16] = 255
        img[img == 15] = 255
        img[img == 14] = 255
        img[img == 13] = 255

        ## Person
        img[img == 11] = 255

        ### Don't need, make these 255
        ## Background
        img[img == 19] = 255


        return img                          
    
    def forward(self, x, base_model, fastscnn): 
        
        ## debug
        # save_image(x[0], 'output-image.png')
        # mask = get_color_pallete(pred_g_merge[0].cpu().numpy())
        # mask.save('output.png')
        sizeH = x.shape[2]
        sizeW = x.shape[3]
        
        # Pad if height or width is odd
        if sizeH % 2 != 0:
            x = F.pad(input=x, pad=(0, 0, 1, 2), mode="constant", value=0)
        if sizeW % 2 != 0:
            x = F.pad(input=x, pad=(1, 2), mode="constant", value=0)

        # Forward pass through fastscnn without gradients
        with torch.no_grad():
            outputs = fastscnn(x)

        if self.visualize:
            # save_image(x[0], 'output-image.png')
            xx = x
            save_batch_images(x)
        
        # Forward pass through base_model
        x = base_model(x)
        N, C, H, W = x.shape
        
        # Initialize graph nodes tensor
        graph_nodes = torch.zeros(N, self.NB, C, H, W).cuda()
        rsizet = transforms.Resize((H, W))
        
        # Process the output of fastscnn to get predicted labels
        pred_all = torch.argmax(outputs[0], 1)
        
        if self.visualize:
            # Assuming `pred_all` is your batch of predictions
            save_batch_masks(pred_all, 'stage2-mask-real.png')
        
        
        pred_all = self.relabel(pred_all)

        if self.visualize:
            # Assuming `pred_all` is your batch of predictions
            save_batch_masks(pred_all, 'stage3-mask-merge.png')
        
        for img_i in range(N):
            all_label_mask = pred_all[img_i]
            labels_all, label_count_all = all_label_mask.unique(return_counts=True)
            ## remove 255 labels
            labels_all = labels_all[:-1]
            label_count_all = label_count_all[:-1]

            # Sort the filtered counts in descending order and get the sorted indices
            sorted_counts, sorted_indices = torch.sort(label_count_all, descending=True)
            
            # Reorder the filtered labels based on the sorted indices
            sorted_labels = labels_all[sorted_indices]
            # Apply the mask after sorting
            mask_t = sorted_counts >= 10000
            labels = sorted_labels[mask_t]


            # Create masks for each label and convert them to bounding boxes
            masks = all_label_mask == labels[:, None, None]
            all_label_mask = rsizet(all_label_mask.unsqueeze(0)).squeeze(0)


            sub_nodes = []
            pre_l2 = x[img_i]
            if self.visualize:
                save_image_with_heatmap(tensor_image=xx[img_i], pre_l2=pre_l2, img_i=img_i)

           
            ### Crop regions
            regions = masks_to_boxes(masks.to(torch.float32))
            boxes = (regions / 16).to(torch.long)
            
            # sub_nodes.append(embed_image.unsqueeze(0))
            for i, _ in enumerate(labels[:min(2, len(labels))]):
                x_min, y_min, x_max, y_max = boxes[i]
                embed_image_c = rsizet(pre_l2[:, y_min:y_max, x_min:x_max])
                if self.visualize:
                    embed_file_name = f'embed_{i}.png'  # Customize the naming pattern as needed
                    x_min, y_min, x_max, y_max = regions[i].to(torch.long)
                    save_image_with_heatmap(tensor_image=xx[img_i][:, y_min:y_max, x_min:x_max], pre_l2=embed_image_c, img_i=img_i, file_name=embed_file_name)
                sub_nodes.append(embed_image_c.unsqueeze(0))

            if len(sub_nodes) < self.NB:
                if self.visualize:
                    save_image_with_heatmap(tensor_image=xx[img_i], pre_l2=pre_l2, img_i=img_i, file_name='pre_l2.png')
                bb_x = [
                    [int(W / 4), int(H / 4), int(3 * W / 4), int(3 * H / 4)],
                    [0, 0, int(2 * W / 3), H],
                    [int(W / 3), 0, W, H],
                    [0, 0, W, int(2 * H / 3)],
                    [0, int(H / 3), W, H]                    
                ]
                for i in range(len(bb_x) - len(sub_nodes)):
                    x_nodes = pre_l2[:, bb_x[i][1]:bb_x[i][3], bb_x[i][0]:bb_x[i][2]]
                    sub_nodes.append(rsizet(x_nodes.unsqueeze(0)))
                    if self.visualize:
                        patch_file_name = f'patch_{i}.png'  # Customize the naming pattern as needed
                        save_image_with_heatmap(tensor_image=xx[img_i], pre_l2=pre_l2, img_i=img_i, file_name=patch_file_name, patch_idx=i)

            # Stack the cropped patches and store them in graph_nodes
            aa = torch.stack(sub_nodes, 1)
            graph_nodes[img_i] = aa[0]

        # Reshape and concatenate graph_nodes with the original tensor x
        x_nodes = graph_nodes.view(self.NB, N, C, H, W)
        x_nodes = torch.cat((x_nodes, x.unsqueeze(0)))
        
        # Clean up
        del graph_nodes, sub_nodes, pred_all, labels_all, label_count_all, masks, all_label_mask
        
        return x.size(0), x_nodes
class GraphVLAD(nn.Module):
    def __init__(self, base_model, net_vlad, fastscnn, NB):
        super(GraphVLAD, self).__init__()
        self.base_model = base_model
        self.fastscnn = fastscnn
        self.net_vlad = net_vlad
        
        self.NB = NB
        self.mask = True
                
        self.applyGNN = applyGNN()
        self.SelectRegions = SelectRegions(self.NB, self.mask)

    def _init_params(self):
        self.base_model._init_params()
        self.net_vlad._init_params()

    def forward(self, x):
        node_features_list = []
        neighborsFeat = []

        x_size, x_nodes = self.SelectRegions(x, self.base_model, self.fastscnn)

        for i in range(self.NB+1):
            vlad_x = self.net_vlad(x_nodes[i])
            # vlad_x = F.normalize(vlad_x, p=2, dim=2)
            # vlad_x = vlad_x.view(x_size, -1)
            # vlad_x = F.normalize(vlad_x, p=2, dim=1)
            neighborsFeat.append(self.net_vlad(x_nodes[i]))
        node_features_list.append(neighborsFeat[self.NB])
        node_features_list.append(torch.concat(neighborsFeat[0:self.NB],0))
        del neighborsFeat
        gvlad = self.applyGNN(node_features_list)
        gvlad = F.normalize(gvlad, p=2, dim=1)

        gvlad = torch.add(gvlad, vlad_x)
        gvlad = F.normalize(gvlad, p=2, dim=1)

        gvlad = gvlad.view(-1, vlad_x.shape[1])
        
        # Clear node_features_list to free up memory
        del node_features_list
        
        return gvlad