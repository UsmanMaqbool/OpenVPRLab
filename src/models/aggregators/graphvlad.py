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

from torch_geometric.data import Data
from torch_geometric.nn import GATConv


from .visualize import get_color_pallete, save_batch_images, save_batch_masks, save_image_with_heatmap,     save_x_nodes_patches


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
        self.input_dim = 256 
        self.hidden_dim = [128,128]
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
        # in_channels = 1024
        # proj_channels = 512
        # reduce input dimension using 3x3 conv
        # self.proj_c = torch.nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        
        # normalize the input to the BoQ blocks
        # self.norm_input = torch.nn.LayerNorm(proj_channels)
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
        # x = self.proj_c(x)
        # x = self.norm_input(x)
        N, C, H, W = x.shape
        
        # Initialize graph nodes tensor
        graph_nodes = torch.zeros(N, self.NB + 1, C, H, W).cuda()
        # graph_nodes = torch.zeros(N, C, H, W).cuda()

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
            
            # # Apply the mask after sorting
            mask_t = sorted_counts >= 10000
            labels = sorted_labels[mask_t]

            # labels = labels_all
            
            # Create masks for each label and convert them to bounding boxes
            masks = all_label_mask == labels[:, None, None]
            all_label_mask = rsizet(all_label_mask.unsqueeze(0)).squeeze(0)

            #resetting the subnodes
            local_rep = []

            pre_l2 = x[img_i]
            # if self.visualize:
            #     save_image_with_heatmap(tensor_image=xx[img_i], pre_l2=pre_l2, img_i=img_i)

           
            ### Crop regions
            regions = masks_to_boxes(masks.to(torch.float32))
            boxes = (regions / 16).to(torch.long)
            
            for i, label in enumerate(labels[:min(5, len(labels))]):
                binary_mask = (all_label_mask == label).float()
                local_feat = x[img_i] * binary_mask
                # local_rep.append(local_feat)
                pre_l2 = local_feat + pre_l2

            # print(f'Number of regions: {len(labels)}')        
            ## Saving the local representation
            for i, _ in enumerate(labels[:min(5, len(labels))]):
                x_min, y_min, x_max, y_max = boxes[i]
                if y_min == y_max or x_min == x_max:
                    continue
                embed_image_c = rsizet(pre_l2[:, y_min:y_max, x_min:x_max])
                # if self.visualize:
                #     embed_file_name = f'embed_{i}.png'  # Customize the naming pattern as needed
                #     x_min, y_min, x_max, y_max = regions[i].to(torch.long)
                #     # save_image_with_heatmap(tensor_image=xx[img_i][:, y_min:y_max, x_min:x_max], pre_l2=embed_image_c, img_i=img_i, file_name=embed_file_name)
                #     save_image_with_heatmap(tensor_image=xx[img_i], pre_l2=embed_image_c, img_i=img_i, file_name=embed_file_name)
                local_rep.append(embed_image_c.unsqueeze(0))

            ## Adding addtional crops is the crops are less
            
            if len(local_rep) < self.NB:
                total_required = self.NB - len(local_rep)
                bb_x = [
                    [int(W / 4), int(H / 4), int(3 * W / 4), int(3 * H / 4)],
                    [0, 0, int(2 * W / 3), H],
                    [int(W / 3), 0, W, H],
                    [0, 0, W, int(2 * H / 3)],
                    [0, int(H / 3), W, H]
                ]
                for i in range(total_required):
                    x_min, y_min, x_max, y_max = bb_x[i]
                    embed_image_c = pre_l2[:, y_min:y_max, x_min:x_max]
                    local_rep.append(rsizet(embed_image_c.unsqueeze(0)))


            local_rep.append(x[img_i].unsqueeze(0)) # store global representation at 5th index.

            graph_nodes[img_i] = torch.stack(local_rep, 1).squeeze(0)

        # Reshape and concatenate graph_nodes with the original tensor x
        x_nodes = graph_nodes.view(self.NB+1, N, C, H, W)
        
        # Clean up
        # del graph_nodes, local_rep, pred_all, labels_all, label_count_all, masks, all_label_mask
        
        # x_nodes = graph_nodes
        
        return x.size(0), x_nodes
    
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GATModel, self).__init__()
        
        self.gat_conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.gat_conv2 = GATConv(out_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat_conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.gat_conv2(x, edge_index)
        return x
class GraphVLAD(nn.Module):
    def __init__(self, base_model, aggregator, fastscnn, NB, edge_index):
        super(GraphVLAD, self).__init__()
        self.base_model = base_model
        self.fastscnn = fastscnn
        self.aggregator = aggregator
        
        self.NB = NB
        self.mask = True
                
        self.applyGNN = applyGNN()
        self.SelectRegions = SelectRegions(self.NB, self.mask)
        
        # Instantiate model
        self.in_channels = 2048  # Feature dimension
        self.out_channels = 2048  # Output dimension of the GAT layer
        self.heads = 1  # Number of attention heads
        self.GATModel = GATModel(self.in_channels, self.out_channels, self.heads)
        
        self.proj_in_channels = 1024
        self.proj_out_channels = 512
        #reduce input dimension using 3x3 conv
        self.proj_c = torch.nn.Conv2d(self.proj_in_channels, self.proj_out_channels, kernel_size=3, padding=1)
        self.proj_l = torch.nn.Conv2d(self.proj_in_channels, 128, kernel_size=3, padding=1)

        self.channel_proj = nn.Linear(self.in_channels, self.proj_in_channels)
        
        # self.edge_index = []
        # for i in range(self.NB):
        #     # Connecting global to local and local to global
        #     self.edge_index.append([self.NB, i])
        #     self.edge_index.append([i, self.NB])
        # self.edge_index = torch.tensor(self.edge_index, dtype=torch.long).t().contiguous()
        # # self.edge_index = edge_index
        self.edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5],  # source nodes
            [5, 5, 5, 5, 5, 5]   # target nodes
            ], dtype=torch.long).contiguous()

        # Move edge_index to the GPU, assuming a GPU is available
        if torch.cuda.is_available():
            self.edge_index = self.edge_index.cuda()


    def _init_params(self):
        self.base_model._init_params()
        self.aggregator._init_params()

    def forward(self, x):
        # xx1 = self.base_model(x)
        # xx2 = self.aggregator(xx1)

        node_features_list = []
        neighborsFeat = []
      
        _, x_nodes = self.SelectRegions(x, self.base_model, self.fastscnn)
        #x_nodes.shape
        #torch.Size([6, 40, 1024, 20, 20])
        
        # l1 = self.proj_c(x_nodes[0])
        # l2 = self.proj_c(x_nodes[1])
        # l3 = self.proj_c(x_nodes[2])
        # l4 = self.proj_c(x_nodes[3])
        # ll = torch.sum(torch.stack([l1, l2, l3, l4]), dim=0)
        # gg = self.proj_c(x_nodes[5])
        
        # lll = torch.cat((gg, ll), dim=1)
        # x = self.aggregator(lll)
        
        for i in range(self.NB+1):
            vlad_x = self.aggregator(x_nodes[i]) # torch.Size([40, 2048])
            neighborsFeat.append(vlad_x)
        # node_features_list.append(neighborsFeat[self.NB])
        # node_features_list.append(torch.concat(neighborsFeat[0:self.NB],0))        
        nodes = torch.stack(neighborsFeat, dim=0)
        feat_size = vlad_x.shape[0] # 40
        nodes = nodes.view(feat_size, self.NB+1,-1)
        for i in range(nodes.shape[0]):
            zz = nodes[i]
            
            ori = zz[5] 
        
            data = Data(x=zz, edge_index=self.edge_index)
            data = self.GATModel(data)
            data = data[5]
        
        
            data2 = self.channel_proj(data)
            ori_2 = self.channel_proj(ori)
            data = torch.cat((data2,ori_2), dim=0) 
            data = data + ori
            data = F.normalize(data, p=2, dim=0)
            node_features_list.append(data)
        
        x = torch.stack(node_features_list, dim=0)
        return x
    
    