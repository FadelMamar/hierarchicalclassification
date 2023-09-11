import rasterio
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from pathlib import Path
from itertools import product
from tqdm import tqdm
import pickle
import os
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import torch
import albumentations as A
from skimage.io import imread
from torch.utils.data import Dataset
import torchvision
from args import Args
import glob
import wandb
import networkx as nx
from itertools import product


# labels of
forest = [1,3,4,5,49] 
farming = [14,15,18,9,21,19,41,39,20,40,36,46,47,48]
nonvegetated=[22,23,24,30,25]
waterbodies=[26,33,31]


def get_coordinates(data,window:Window):

    x_min, y_min, x_max, y_max = data.window_bounds(window) 

    return x_min, y_min, x_max, y_max

def get_tile_label(unique_vals:list,unique_counts:list,threshold=0.6):

    sum_counts = sum(unique_counts)
    label = -999

    for ind in range(len(unique_vals)):
        if unique_counts[ind]/sum_counts >= 0.6:
            label = unique_vals[ind]
            break

    return label

def parse_geotif(labels_path:Path,tilesize=38):

    #labels_path = Path(f"../data/brasil_coverage_{year}.tif")
    data = rasterio.open(labels_path)

    height, width = data.meta['height'],data.meta['width']
    height,width
    labels_dict = dict()
    #tilesize=38 # ~1km*

    print('...get labels distribution per tile')
    for i,j in tqdm(product(list(range(0,height,tilesize)),list(range(0,width,tilesize)))):
        window = Window(j, i, tilesize, tilesize)
        chunk = data.read(1,window=window)
        unique_vals,unique_counts = np.unique(chunk,return_counts=True)
        if unique_vals.shape[0] != 1:  
            labels_dict[(i,j)] = (unique_vals.tolist(),unique_counts.tolist(),window)

    save_path=f'../data/{os.path.basename(labels_path).split(sep=".")[0]}_labelDist_2.pickle'
    print('saving dictionary as pickle at: ',save_path)
    with open(save_path,'wb') as file:
        pickle.dump(labels_dict,file)
    
    print('Number of tiles: ',len(labels_dict))
    data.close()

    return labels_dict,save_path

def get_labeldistribution(parsed_tiles:dict,labels_path:Path,majority_threshold=0.6):

    data = rasterio.open(labels_path)

    # # # Get distribution of labels
    cols = ["block_x","block_y","xmin","ymin","xmax","ymax","tile_label"] #+ ['label-'+str(a) for a in all_labels]
    labels_dist = dict()
    ind = 0
    
    print('...create csv for labels distribution per tile')

    for position,(unique_vals,unique_counts,window) in tqdm(parsed_tiles.items()):

        block_y,block_x = position
        x_min, y_min, x_max, y_max = get_coordinates(data=data,window=window) 
        label = get_tile_label(unique_vals,unique_counts,
                               threshold=majority_threshold)
        
        labels_dist[ind] = [block_x,block_y,x_min,y_min,x_max,y_max,label] 
        ind += 1
    
    df_labelsDist = pd.DataFrame.from_dict(labels_dist, orient='index',columns=cols)
    save_path=f'../data/{os.path.basename(labels_path).split(sep=".")[0]}_labelDist_2.csv'
    df_labelsDist.to_csv(save_path,index=False)
    print('csv file saved at:',save_path)

    data.close()

    return df_labelsDist,save_path

def get_split(block_indices:list,schema,num_blocks_per_tile=1):
    
    x,y = block_indices
    height,width = schema.shape

    split = schema[(y//num_blocks_per_tile)%height,
                   (x//num_blocks_per_tile)%width]

    return split

def train_test_split(label_distribution:pd.DataFrame,
                     num_blocks_per_tile=5,
                     splits_by_block=None,
                     labeling_schema=None):
        
    label_distribution['train_test_split'] = -999 # no split attributed

    if splits_by_block is None:

        # define labeling schema
        if labeling_schema is None:
            labeling_schema = schema = np.array([[0,2,1,0,0],
                                                [2,1,0,0,0],
                                                [1,0,0,0,2],
                                                [0,0,0,2,1]])
        
        tilesize=38 # 1km*1km
        #-- get indexing of blocks   
        label_distribution['block_x_index'] = ((label_distribution.block_x - label_distribution.block_x.min()) / tilesize).apply(int)
        label_distribution['block_y_index'] = ((label_distribution.block_y - label_distribution.block_y.min()) / tilesize).apply(int)

        assert label_distribution.duplicated(['block_x','block_y']).sum() < 1, "Some duplicates are present! Check them out."

        #-- resetting indexing of DataFrame
        label_distribution = label_distribution.set_index(['block_x_index','block_y_index'],inplace=False)

        for x,y in tqdm(label_distribution.index):
            split = get_split(block_indices=(x,y),
                            schema=labeling_schema,
                            num_blocks_per_tile=num_blocks_per_tile)
            label_distribution.at[(x,y),'train_test_split'] = split
        
        label_distribution.reset_index(inplace=True) # resetting

    #-- too slow
    else:
        assert isinstance(splits_by_block,pd.DataFrame),'Provide a DataFrame'
        assert len(set(["xmin","ymin","xmax","ymax","train_test_split"]) & set(splits_by_block.columns)) == len(["xmin","ymin","xmax","ymax","train_test_split"]), 'Required columns are not available. Please check splits_by_block.'
        for row in tqdm(splits_by_block.itertuples()):
            mask_x = (label_distribution.xmin >= row.xmin) & (label_distribution.xmax <= row.xmax) 
            mask_y = (label_distribution.ymin >= row.ymin) & (label_distribution.xmax <= row.ymax)  
            label_distribution.loc[mask_x & mask_y,'train_test_split'] = row.train_test_split
        
    return label_distribution

def get_samplingweight(x,y,labelling_schema):

    height,width = labelling_schema.shape

    y,x = y%height,x%width

    prob_y = norm.pdf(y,loc=height/2,scale=height/4)
    prob_x = norm.pdf(x,loc=width/2,scale=width/4)    

    return 0.5*(prob_y+prob_x)

def get_targetdistribution(labelDistribution:pd.DataFrame,min_examples=100,ratio=3):

    labelDistribution = labelDistribution[labelDistribution.tile_label > 0] # get rid of -999 and 0

    target_distribution = labelDistribution.groupby(by=['train_test_split','tile_label']).count()[['xmin']]
    target_distribution['to_sample'] = 0

    # compute number of tiles to sample
    for split,tile_label in target_distribution.index:

        count = target_distribution.at[(split,tile_label),'xmin']
        if count < min_examples:
            target_distribution.at[(split,tile_label),'to_sample'] = count

        elif split in [1,2]:
            target_distribution.at[(split,tile_label),'to_sample'] = min_examples

        elif split == 0:
            target_distribution.at[(split,tile_label),'to_sample'] = min(min_examples*ratio,count)    

    # drop 'xmin' and reset indexing
    target_distribution = target_distribution[['to_sample']].reset_index()

    # get counts per split
    splits = target_distribution.groupby(by='train_test_split').sum()[['to_sample']]
    splits['percentage'] = 100*splits.to_sample / splits.to_sample.sum()
    
    # normalize distribution
    target_distribution['percentage'] = target_distribution['to_sample']
    for split in [0,1,2]:
        target_distribution.loc[target_distribution.train_test_split == split,'percentage'] =   100*target_distribution.loc[target_distribution.train_test_split == split,'percentage']/splits.at[split,'to_sample']


    print('Sampled data splits: \n',splits)

    return target_distribution

def get_sample(target_distribution:pd.DataFrame,
                    source_data:pd.DataFrame,
                    labelling_schema,plot=True):

    print('Labelling schema: \n', labelling_schema )

    # copy
    data = source_data.copy(deep=True)
    target = target_distribution.copy(deep=True)
    
    data = data[data.tile_label > 0] # get rid of -999 and 0

    # compute sampling weights
    data['sampling_weight'] = get_samplingweight(data.block_x_index,
                                                data.block_y_index,
                                                labelling_schema=labelling_schema)

    # reset indexing
    target.set_index(keys=['train_test_split','tile_label'],inplace=True)
    data['selected'] = False

    # select data
    print('selecting tiles:')
    for split,label in tqdm(target.index):
        mask = (data.train_test_split == split) & (data.tile_label == label) & (data.selected == False)
        target_count = target.at[(split,label),'to_sample']
        selected_indices = data.loc[mask].sample(target_count,weights='sampling_weight',random_state=split+label).index  
        data.loc[data.index.isin(selected_indices),'selected'] = True
    
    target.reset_index(inplace=True)
    
    # plot distributions
    if plot:
        fig,axs = plt.subplots(3,1,figsize=(15,20))
        sns.countplot(data=data,x='tile_label',hue='train_test_split',ax=axs[0])
        axs[0].set_title('Source data: label distribution')
        axs[0].set_yscale('log')

        sns.countplot(data=data[data.selected == True],x='tile_label',hue='train_test_split',ax=axs[1])
        axs[1].set_title('Sampled data: label distribution')

        sns.barplot(data=target,x='tile_label',y='percentage',hue='train_test_split',ax=axs[2])
        axs[2].set_title('Sampled data: normalized label distribution')

    return data[data.selected == True]

def get_dataset_labels(label_hierarchy_path:Path) -> np.ndarray:

    label_hierarchy = pd.read_csv(label_hierarchy_path,sep=';')
    all_labels = []

    for col in label_hierarchy.columns:
        all_labels = all_labels + label_hierarchy[col].unique().tolist()
    
    all_labels = np.array(sorted(list(set(all_labels)),reverse=False))

    return all_labels

def get_label(all_labels:np.ndarray,leafnode_label:int,labelhiearchy:pd.DataFrame) -> np.ndarray:
    
    if leafnode_label in labelhiearchy.index:
        path_in_hierarchy = set(labelhiearchy.loc[leafnode_label].to_list())

    else: # parent node != leaf
        path_in_hierarchy = [leafnode_label]

    all_labels = all_labels
    
    # build binary label
    labels = np.zeros_like(all_labels)

    for l in path_in_hierarchy:
        labels[np.where(all_labels == l)] = 1
   
    return labels

def get_level_indices(label_hierarchy_path:Path):

    #- load labels
    all_labels = get_dataset_labels(label_hierarchy_path)
    label_hierarchy = pd.read_csv(label_hierarchy_path,sep=';')

    level1 = np.sort(label_hierarchy['level1'].unique()).tolist()
    level2 = np.sort(label_hierarchy['level2'].unique()).tolist()
    level3 = np.sort(label_hierarchy['level3'].unique()).tolist()
    level4 = np.sort(label_hierarchy['level4'].unique()).tolist()
    leaf_nodes = np.sort(label_hierarchy['leaf'].unique()).tolist()
    level3 = [a for a in level3 if a not in level2] 
    level4 = [a for a in level4 if a not in level2]

    indices_level1 = np.hstack([np.where(all_labels == a)[0] for a in level1])
    indices_level2 = np.hstack([np.where(all_labels == a)[0] for a in level2])
    indices_level3 = np.hstack([np.where(all_labels == a)[0] for a in level3])
    indices_level4 = np.hstack([np.where(all_labels == a)[0] for a in level4])
    indices_leaf = np.hstack([np.where(all_labels == a)[0] for a in leaf_nodes])

    #- checks
    levels = [level1,level2,level3,level4]
    assert sum([len(l) for l in levels]) == all_labels.shape[0],'Mismtach!'
    for Li, Lj in product(levels,levels):
        assert float(np.intersect1d(Li,Lj).shape[0]) in [0.0,(len(Li)+len(Lj))/2]

    return indices_level1,indices_level2,indices_level3,indices_level4,indices_leaf

def build_hierarchy_graph(label_hierarchy_path:Path=Path("../data/labelhierarchy.csv"),
                          nodeLabel_as_target=False):
    
    all_labels = get_dataset_labels(label_hierarchy_path)
    label_hierarchy = pd.read_csv(label_hierarchy_path,sep=';')

    # define node label format
    if nodeLabel_as_target:
        mapping_func = lambda x: tuple(get_label(all_labels,x,label_hierarchy.set_index('leaf')).tolist())
    else:
        mapping_func = lambda x: x
    
    # Create nad populate graph
    graph = nx.Graph()
    nodes = [(mapping_func(a), {'label':a}) for a in all_labels]
    graph.add_nodes_from(nodes)

    cols= label_hierarchy.columns.to_list()
    for i in range(1,len(cols)-1):
        level_i,level_j = cols[i],cols[i+1]
        edges = list(zip(label_hierarchy[level_i].to_list(),label_hierarchy[level_j].to_list()))
        edges = [(mapping_func(a),mapping_func(b)) for a,b in edges if a!=b]
        graph.add_edges_from(edges)

    # add root node and its edges
    graph.add_node(0)
    graph.add_edges_from([(0,mapping_func(1)),(0,mapping_func(10)),
                          (0,mapping_func(14)),(0,mapping_func(22)),
                          (0,mapping_func(26))])

    return graph,mapping_func

def tree_distance(graph:nx.Graph,
                  source,destination,
                  mapping_func=lambda x:x):

    d = len(nx.shortest_path(graph,
                         mapping_func(source),
                         mapping_func(destination))) - 1
    return d

def reverse_label_encoding(onehot_label:np.ndarray,all_labels:np.ndarray) -> np.ndarray:

    assert all_labels.shape[0] in onehot_label.shape

    return all_labels.reshape((1,-1)) * onehot_label

def get_ancestry_matrix(label_hierarchy_path : Path = Path("../data/labelhierarchy.csv")):

    label_hierarchy = pd.read_csv(label_hierarchy_path,sep=';')
    # label_hierarchy.set_index('leaf',inplace=True)
    all_labels = get_dataset_labels(label_hierarchy_path)
    # build ancestry matrix A
    # A[i,j] = 1 if "j" is subclass of "i"
    n = all_labels.shape[0] # number of labels
    A = np.zeros((n,n))
    indices = {label:np.where(all_labels==label)[0] for label in all_labels}

    for label in all_labels:
        # iterate through hierarchy
        hierarchy = label_hierarchy.loc[label_hierarchy.leaf == label]
        assert len(hierarchy) < 2, 'Please check the csv file @ ``data/labelhierarchy.csv``'
        if len(hierarchy)>0:
            for i in hierarchy.iloc[0].to_list():
                A[indices[i],indices[label]] = 1.0
        # set every label as it's own descendent
        A[indices[label], indices[label]] = 1.0

    return A,indices

def transform_data(
        image: np.ndarray,
        args:Args,
        apply_transform: bool):
    """transform function used in Dataloader.

    Args:
        image (np.ndarray): b-scan
        label (np.ndarray): raw manual annotation
        args (Args): arguments parsed from Args.py which contains the configuration of the experiment.
        apply_transform (bool): states if data augmentation should be enabled or not
        layer_segmentation (np.ndarray,None): layer segmentation from Discovery
        is_valid (bool, optional): It should be set to False only for training. Defaults to False.

    Raises:
        NotImplementedError: raised when args.resizing_mode is not in ['nearest','None'].

    Returns:
        tuple: returns image and target mask as torch.tensor
    """
        
    TRANSFORMS = {
        "GaussianBlur": A.augmentations.GaussianBlur(
            p=0.5),
        "HorizontalFlip": A.HorizontalFlip(
            p=0.5),
        "ShiftScaleRotate": A.augmentations.ShiftScaleRotate(
            p=0.5,
            rotate_limit=args.rotate_degree)
        }

    # Normalization mean and std
    mean,std = args.traindata_mean_4channels,args.traindata_std_4channels

    # Cast to 3 channels
    # compute NDVI
    assert args.in_channels in [3,4],'provide a correct number of channels. Either 3 or 4.'
    if args.in_channels == 3:
        ndvi = (image[:,:,3] - image[:,:,0])/(image[:,:,3] + image[:,:,0] + 1e-8)
        image[:,:,0] = ndvi
        image = image[:,:,:-1] # drop Nir
        mean,std = args.traindata_mean,args.traindata_std


    # -- Get transforms
    transform = A.Compose([TRANSFORMS[t] for t in args.augmentations])

    if apply_transform:
        transformed = transform(image=image)        
        image = torch.Tensor(transformed['image'].astype(float)).transpose(0,2)

    else:
        image = torch.Tensor(image.astype(float)).transpose(0,2)

    # -- standardized
    normalizer = torchvision.transforms.Normalize(mean=mean,
                                                  std=std)
    if args.apply_augmentation:
        image = normalizer(image)

    return  image

class Mydataset(Dataset):

    def __init__(self, mode, args:Args):
        """2D dataset constructor

        Args:
            mode (str): either 'train' or 'valid'.
            args (Args): arguments parsed from Args.py which contains the configuration of the experiment.
        """
        self.mode = mode
        self.is_valid = mode != 0
        self.args = args
        self.data_dir = self.args.data_dir
        
        #-- flatten label hierarchy
        self.label_hierarchy = pd.read_csv(args.label_hierarchy,sep=';')
        self.label_hierarchy.set_index('leaf',inplace=True)
        self.all_labels = get_dataset_labels(args.label_hierarchy)

        #-- get dataset info
        self.data_info = pd.read_csv(self.args.data_info,sep=',')
        self.data_info = self.data_info[self.data_info['tile_label'] != 32]
        self.data_info = self.data_info[self.data_info.train_test_split == mode]
        
        #-- get number of images in dataset
        self.images = [list((self.data_dir/str(id_)).glob('*.tif'))  for id_ in self.data_info.sampleid]  # -- image filenames
        self.num_images = len(self.images)
        
        #-- get transform function
        self.apply_transform = self.args.apply_augmentation and (self.mode == 0)  
        self.transform = transform_data

    def __len__(self):
        return self.num_images

    def get_image(self, idx):
        """Gets an image and its mask provided an index

        Args:
            idx (int): index of the sample

        Returns:
            tuple: (image,label)
        """

        # get image
        file_name = self.images[idx][0]
        assert len(self.images[idx]) == 1, 'there should be only one tif in a folder'
        image = imread(file_name)
        image = image[:,:,[2,1,0,3]] # get as R-G-B-Nir

        # transform image
        image = self.transform(image, self.args, self.apply_transform)

        # get label
        sampleid = int(file_name.parent.name)
        leafnodelabel = self.data_info.loc[self.data_info.sampleid == sampleid,'tile_label'].iat[0]
        label = get_label(self.all_labels,
                          leafnodelabel,
                          self.label_hierarchy)

        return image, label

    def __getitem__(self, idx):
        """Return item and also runs a mix-up operation is required.

        Args:
            idx (int): index of the sample

        Returns:
            tuple: (image,mask)
        """
        image, label = self.get_image(idx)

       
        return image, label

def log_batch_to_wandb(
        batch_img:torch.Tensor,
        batch_true_label:np.ndarray,
        batch_pred_label:np.ndarray,
        stage: str,
        epoch: int,
        step: int,
        args,
        attention_map=None):


    #-- unnormalize
    mean = torch.Tensor(args.traindata_mean_4channels)
    std = torch.Tensor(args.traindata_std_4channels)

    if args.in_channels == 3:
        mean = torch.Tensor(args.traindata_mean)
        std = torch.Tensor(args.traindata_std)

    std_inv = 1 / (std + 1e-7)
    unnormalize = torchvision.transforms.Normalize(-mean * std_inv, std_inv)
    
    batch_img = unnormalize(batch_img)
    batch_img = batch_img[:,:3,:,:].transpose(1,3) # if 4-channels, last channels Nir is dropped

    # change: Tensor -> numpy
    batch_img =  batch_img.numpy()/4e3

    # -- plot
    if attention_map is not None and stage == 'valid':
        saliency = attention_map

    # Let's log 20 sample image predictions from the first batch
    n = min(15,batch_img.shape[0])
    columns = ['image', 'ground truth', 'prediction']
    data = [[wandb.Image(x_i),
             str(y_i[y_i.nonzero()]),
             str(y_pred[y_pred.nonzero()])] for x_i, y_i, y_pred in list(zip(batch_img[:n],
                                                                            batch_true_label[:n],
                                                                            batch_pred_label[:n]))]
    wandb.log({f'Images_{stage}_epoch:{epoch}':wandb.Table(columns=columns, data=data)})            


    return None