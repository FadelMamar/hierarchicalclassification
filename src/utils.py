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
    """Get data window bounds

    Args:
        data (rasterio.DatasetReader): GeoTiFF reading handler from Rasterio
        window (Window): window of data to return

    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, x_max, y_max = data.window_bounds(window) 

    return x_min, y_min, x_max, y_max

def get_tile_label(unique_vals:list,unique_counts:list,threshold=0.6):
    """Get tile label. Assign the majority label within a tile given a desired threshold.

    Args:
        unique_vals (list): labels in tile
        unique_counts (list): counts of each label
        threshold (float, optional): thresolding. Defaults to 0.6.

    Returns:
        int: label assigned to tile
    """

    sum_counts = sum(unique_counts)
    label = -999

    for ind in range(len(unique_vals)):
        if unique_counts[ind]/sum_counts >= threshold:
            label = unique_vals[ind]
            break

    return label

def parse_geotif(labels_path:Path,tilesize=38):
    """Read a GeoTiFF and save tile labels values and counts in a pickle file as a dictionary ``labels_dict``.

    Args:
        labels_path (Path): path to GeoTiFF file
        tilesize (int, optional): height and width of tile in pixels. Defaults to 38.

    Returns:
        tuple: (labels_dict,save_path)
    """
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
    """Get label distribution within a GeoTiFF. Returns the labels distribution in a pd.DataFrame and the path ``save_path`` where it was saved.

    Args:
        parsed_tiles (dict): dictionary of labels counts and values computed by ``parse_geotif``
        labels_path (Path): path to GeoTiFF
        majority_threshold (float, optional): _description_. Defaults to 0.6.

    Returns:
        tuple: (label distribution in pd.DataFrame save_path)
    """
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

def get_split(block_indices:list,schema:np.ndarray,num_blocks_per_tile=1):
    """Assign a block to the training, validation or test set. A block is a portion of a tile.

    Args:
        block_indices (list): x,y indices of a block
        schema (np.ndarray): splitting schema
        num_blocks_per_tile (int, optional): number of blocks per tile. Defaults to 1.

    Returns:
        str: split value
    """
    x,y = block_indices
    height,width = schema.shape

    split = schema[(y//num_blocks_per_tile)%height,
                   (x//num_blocks_per_tile)%width]

    return split

def train_test_split(label_distribution:pd.DataFrame,
                     num_blocks_per_tile=5,
                     splits_by_block=None,
                     labeling_schema=None):
    """Assign every block to a training, validation or test set

    Args:
        label_distribution (pd.DataFrame): label distribution computed by ``get_labeldistribution``
        num_blocks_per_tile (int, optional): number of blocks per tile. Defaults to 5.
        splits_by_block (pd.DataFrame, optional): pd.DataFrame with block bounds and assigned split. Defaults to None.
        labeling_schema (np.ndarray, optional): labeling schema defined as a matrix. Defaults to None.

    Returns:
        pd.DataFrame: contains block bounds and assign set
    """
    label_distribution['train_test_split'] = -999 # no split attributed

    if splits_by_block is None:

        # define labeling schema
        if labeling_schema is None:
            labeling_schema = np.array([[0,2,1,0,0],
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

def get_samplingweight(x:int,y:int,labelling_schema:np.ndarray):
    """Compute sampling weight for every block in a set. It computes from the PDF of a gaussina distribution centered on the tile

    Args:
        x (int): x index of block in a tile
        y (int): y index of block in a tile
        labelling_schema (np.ndarray): labeling schema

    Returns:
        _type_: _description_
    """
    height,width = labelling_schema.shape

    y,x = y%height,x%width

    prob_y = norm.pdf(y,loc=height/2,scale=height/4)
    prob_x = norm.pdf(x,loc=width/2,scale=width/4)    

    return 0.5*(prob_y+prob_x)

def get_targetdistribution(labelDistribution:pd.DataFrame,min_examples=100,ratio=3):
    """Computes the distributin of the wanted training, validation and test sets used for model training and evaluation.

    Args:
        labelDistribution (pd.DataFrame): label distribution from ``train_test_split``
        min_examples (int, optional): minimum nuber of desired samples per label. Defaults to 100.
        ratio (int, optional): set how big the training set should be compared to validation and test sets. Defaults to 3 to have 60/20/20 splits.

    Returns:
        pd.DataFrame: target distribution
    """
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
                    labelling_schema:np.ndarray,
                    plot=True):
    """Samples data from a source label distribution given a target distribution

    Args:
        target_distribution (pd.DataFrame): target distribution
        source_data (pd.DataFrame): source distribution
        labelling_schema (np.ndarray): labelling schema
        plot (bool, optional): states whether or not to plot the distributions. Defaults to True.

    Returns:
        pd.DataFrame: sampled data
    """
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
    """Get all labels within the dataset

    Args:
        label_hierarchy_path (Path): path to label hierarchy csv

    Returns:
        np.ndarray: labels of land use and  land cover classes
    """
    label_hierarchy = pd.read_csv(label_hierarchy_path,sep=';')
    all_labels = []

    for col in label_hierarchy.columns:
        all_labels = all_labels + label_hierarchy[col].unique().tolist()
    
    all_labels = np.array(sorted(list(set(all_labels)),reverse=False))

    return all_labels

def get_label(all_labels:np.ndarray,leafnode_label:int,labelhiearchy:pd.DataFrame) -> np.ndarray:
    """Get the labels on a given root-to-node path defined by the leafnode label. It follwos the label hierarchy graph

    Args:
        all_labels (np.ndarray): all labels from ``get_dataset_labels``
        leafnode_label (int): leaf node with no descendants
        labelhiearchy (pd.DataFrame): label hierarchy indexed on the leaf-label nodes

    Returns:
        np.ndarray: binary array characterizing the labels on a given a roo-to-node path
    """
    if leafnode_label in labelhiearchy.index:
        path_in_hierarchy = set(labelhiearchy.loc[leafnode_label].to_list())

    else: # no parent
        path_in_hierarchy = [leafnode_label]

    all_labels = all_labels
    
    # build binary label
    labels = np.zeros_like(all_labels)

    for l in path_in_hierarchy:
        labels[np.where(all_labels == l)] = 1
   
    return labels

def get_level_indices(label_hierarchy_path:Path):
    """Get indices of labels at every level of the label hierarchy graph

    Args:
        label_hierarchy_path (Path): path to label hierarchy csv

    Returns:
        tuple: indices_level1, indices_level2, indices_level3, indices_level4, indices_leaf
    """
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
    """Builds a label hierarchy graph

    Args:
        label_hierarchy_path (Path, optional): Defaults to Path("../data/labelhierarchy.csv").
        nodeLabel_as_target (bool, optional): states whether the nodes should be encoded using their root-to-node path. Defaults to False.

    Returns:
        tuple: (graph, label mapping function)
    """
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
                  source:int,
                  destination:int,
                  mapping_func=lambda x:x):
    """Tree distance defined as the number of edges in the shortest path

    Args:
        graph (nx.Graph): label hierarchy graph
        source (int): source node
        destination (int): destination node
        mapping_func (function, optional): a function that maps a node label to its encoding in the graph. Defaults to lambdax:x.

    Returns:
        _type_: _description_
    """
    d = len(nx.shortest_path(graph,
                         mapping_func(source),
                         mapping_func(destination))) - 1
    return d

def reverse_label_encoding(onehot_label:np.ndarray,all_labels:np.ndarray) -> np.ndarray:
    """Revert the onehot label encoding

    Args:
        onehot_label (np.ndarray): label encoding
        all_labels (np.ndarray): all labels in the dataets from ``get_dataset_labels``

    Returns:
        np.ndarray: _description_
    """
    assert all_labels.shape[0] in onehot_label.shape

    return all_labels.reshape((1,-1)) * onehot_label

def get_ancestry_matrix(label_hierarchy_path : Path = Path("../data/labelhierarchy.csv")):
    """Get ancestry matrix of the label hierarchy graph. It provides the ancestors of every label in the hierarchy.

    Args:
        label_hierarchy_path (Path, optional): path to label hierarchy. Defaults to Path("../data/labelhierarchy.csv").

    Returns:
        tuple: (ancestry matrix, {label:onehot encoding} )
    """
    label_hierarchy = pd.read_csv(label_hierarchy_path,sep=';')
    all_labels = get_dataset_labels(label_hierarchy_path)
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
    """Transform function used in Dataloader.

    Args:
        image (np.ndarray): satellite image
        args (Args): arguments parsed from args.py which contains the configuration of the experiment.
        apply_transform (bool): states if data augmentation should be enabled or not

    Returns:
        torch.Tensor: transformed image
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

