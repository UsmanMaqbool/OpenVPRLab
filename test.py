import sys
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import importlib
from src.core.vpr_datamodule import VPRDataModule
from src.core.vpr_framework import VPRFramework
from tqdm import tqdm
from gsvcities.utils.validation import get_validation_recalls
# from dataloaders.val.CrossSeasonDataset import CrossSeasonDataset
# from dataloaders.val.EssexDataset import EssexDataset
# from dataloaders.val.InriaDataset import InriaDataset
# from dataloaders.val.NordlandDataset import NordlandDataset
# from dataloaders.val.SPEDDataset import SPEDDataset
from gsvcities.dataloaders.val.MapillaryDataset import MSLS
from gsvcities.dataloaders.val.PittsburghDataset import PittsburghDataset

# from main import VPRModel

def load_config(config_path='model_config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_instance(module_name, class_name, params):
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(**params)


# Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IM_SIZE = (320, 320)
BATCH_SIZE = 40
# VAL_DATASET_NAMES = ['CrossSeason', 'Essex', 'Inria', 'MSLS', 'SPED', 'Nordland', 'pitts30k_test', 'pitts250k_test']
VAL_DATASET_NAMES = ['MSLS', 'pitts250k_test', 'pitts30k_test']

# Functions
def input_transform(image_size=IM_SIZE):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def get_val_dataset(dataset_name, input_transform=input_transform()):
    dataset_name = dataset_name.lower()
    
    if 'cross' in dataset_name:
        ds = CrossSeasonDataset(input_transform=input_transform)
    elif 'essex' in dataset_name:
        ds = EssexDataset(input_transform=input_transform)
    elif 'inria' in dataset_name:
        ds = InriaDataset(input_transform=input_transform)
    elif 'nordland' in dataset_name:
        ds = NordlandDataset(input_transform=input_transform)
    elif 'sped' in dataset_name:
        ds = SPEDDataset(input_transform=input_transform)
    elif 'msls' in dataset_name:
        ds = MSLS(input_transform=input_transform)
    elif 'pitts' in dataset_name:
        ds = PittsburghDataset(which_ds=dataset_name, input_transform=input_transform)
    else:
        raise ValueError("Unknown dataset name")
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth

def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating descriptors...'):
            imgs, labels = batch
            output = model(imgs.to(device)).cpu()
            descriptors.append(output)
    return torch.cat(descriptors)

# Main script
if __name__ == "__main__":
    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define device
    from argparser import parse_args
    config = parse_args()
        
    # Create backbone
    backbone = get_instance(config['backbone']['module'], config['backbone']['class'], config['backbone']['params'])
    out_channels = backbone.out_channels  # all backbones should have an out_channels attribute

    # Update aggregator's in_channels if necessary
    if 'in_channels' in config['aggregator']['params']:
        if config['aggregator']['params']['in_channels'] is None:
            config['aggregator']['params']['in_channels'] = out_channels

    # Create aggregator
    aggregator = get_instance(config['aggregator']['module'], config['aggregator']['class'], config['aggregator']['params'])

    if 'graphvlad' in config['aggregator']:
        segmentation = get_instance(config['segmentation']['module'], config['segmentation']['class'], config['segmentation']['params'])
    else:
        segmentation = None
        
    loss_function = get_instance(config['loss_function']['module'], config['loss_function']['class'], config['loss_function']['params'])


    # Load model
    vpr_model = VPRFramework(
        backbone=backbone,        
        segmentation=segmentation,
        loss_function=loss_function,
        aggregator=aggregator,
        config_dict=config, # pass the config to the framework in order to save it
    )
    
    # state_dict = torch.load('../LOGS/best_models/resnet50_ConvAP_512_2x2.ckpt')
    # Add argument for loading state_dict
    # parser = argparse.ArgumentParser(description='VPR Model Evaluation')
    # parser.add_argument('--load_state_dict', type=str, help='Path to the model state_dict file')
    # args = parser.parse_args()

    # Load state_dict from the provided path
    state_dict_path = config["test"]["load_state_dict"]
    if state_dict_path:
        state_dict = torch.load(state_dict_path)
        vpr_model.load_state_dict(state_dict['state_dict'])
    else:
        raise ValueError("Please provide a valid path to the model state_dict file using --load_state_dict")
    vpr_model.eval()
    model = vpr_model.to(device)

    # Evaluate on all benchmarks
    for val_name in VAL_DATASET_NAMES:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name)
        val_loader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)
        print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, device)
        
        print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[:num_references]
        q_list = descriptors[num_references:]
        
        recalls_dict, preds = get_validation_recalls(r_list=r_list,
                                                    q_list=q_list,
                                                    k_values=[1, 5, 10, 15, 20, 25],
                                                    gt=ground_truth,
                                                    print_results=True,
                                                    dataset_name=val_name,
                                                    faiss_gpu=False)
        del descriptors
        print('========> DONE!\n\n')