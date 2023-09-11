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
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_samplingweight,train_test_split,get_targetdistribution,get_sample
from shapely.geometry import Point

if __name__ == "__main__":
    # Labels
    forest = [3,4,5,49] 
    natural_nonforest = [11,12,32,29,13]
    farming = [15,9,21,41,39,20,40,46,47,48]
    nonvegetated=[23,24,30,25]
    waterbodies=[33,31]

    all_labels = forest + farming + nonvegetated + waterbodies + natural_nonforest
    labels_mapping = {'forest':forest,'farming':farming,'nonvegetated':nonvegetated,'waterbodies':waterbodies,'natural-nonforest':natural_nonforest}


    # Get data
    year=2018

    labels_path = Path(f"../data/brasil_coverage_{year}.tif")
    df_labelsDist2 = pd.read_csv(f'../data/{os.path.basename(labels_path).split(sep=".")[0]}_labelDist_2.csv')

    # splitting schema
    schema = np.array([[0,2,1,0,0],
                    [2,1,0,0,0],
                    [1,0,0,0,2],
                    [0,0,0,2,1]])

    # split data
    df_labelsDist2 = train_test_split(df_labelsDist2,
                                    num_tiles_per_block=5,
                                    labeling_schema=schema)

    #-- check if all labels appear in split
    intersection = list(set(df_labelsDist2.tile_label.unique().tolist()) & set(all_labels))
    missing_labels = [a for a in all_labels if a not in intersection]
    print('the missing labels are', missing_labels)

    # get target distribution
    target_distribution = get_targetdistribution(df_labelsDist2,min_examples=150,ratio=3)

    # sample data
    df_sample = get_sample(target_distribution,df_labelsDist2,schema)

    # compute centroid of tiles
    df_sample['x_center'] = 0.5*(df_sample.xmin + df_sample.xmax)
    df_sample['y_center'] = 0.5*(df_sample.ymin + df_sample.ymax)

    # create Point object
    df_sample['geometry'] = df_sample.apply(lambda row: Point(row.x_center, row.y_center), axis=1)

    # create sampleid column
    df_sample = df_sample.reset_index().rename(columns={'index':'sampleid'})

    #-- save file as csv
    save_path = f'../data/{os.path.basename(labels_path).split(sep=".")[0]}_sample.csv'
    df_sample.to_csv(save_path,index=False)