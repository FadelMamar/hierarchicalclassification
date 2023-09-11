
'''
This script downloads images from Earth Engine according to given parameters.



Example of use:

python download_images.py   --save_path ../data/images/new\
    --sensor S2\
    --sample_points_path ../data/labels/campaign_labels.shp\
    --start_date 2016-01-01\
    --end_date 2023-01-0\
    --num_workers 32 \
    --period_length 1M\
    --buffer 750\
    --processing_level TOA \
    --target_shape '(151,151)'

'''
import utm
import datetime
from geetools import cloud_mask
import argparse
import csv
import json
from multiprocessing.dummy import Pool, Lock
import os
from collections import OrderedDict
import time
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import warnings
warnings.simplefilter('ignore', UserWarning)
import pdb

import pandas as pd
import ee
import numpy as np
import rasterio
import urllib3
from rasterio.transform import Affine
from skimage.exposure import rescale_intensity
import geopandas as gpd
from rasterio.enums import Resampling


#Adding the parent directory to sys.path so that I can import from it
# import sys
# sys.path.append(os.getcwd())
#sys.path.append('/home/jan/Documents/EPFL/drivers')
#from utils import BANDS

LAST_IDX_LOGFILE = "../data/last_idx.txt"


BANDS = {
    "S2" : ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12', 'QA60'],
    "S2_RGBNIR": ['B2', 'B3', 'B4', 'B8', 'QA60'], #These are the bands I am currently downloading 
    #"L8" : ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', "B11", "QA_PIXEL"] # this is TOA
    "L8" : ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'] # this is SR
}


class GeoSampler:

    def sample_point(self):
        raise NotImplementedError()


class ShapeFileSampler(GeoSampler):

    def __init__(self, shapefile_path):
        try:
            self.points = gpd.read_file(shapefile_path)
        except:
            df = pd.read_csv(shapefile_path)
            df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
            self.points = gpd.GeoDataFrame(df)

        self.start_index = 0
        self.end_index = self.points.index[-1]
        
        if os.path.exists(LAST_IDX_LOGFILE):
            self.index = int(open(LAST_IDX_LOGFILE).read())
            self.actual_started_index = self.index
            print(f"starting from idx {self.index} from {LAST_IDX_LOGFILE}")
        else:
            self.index = self.points.index[0]
            self.actual_started_index = self.points.index[0]

        self.lock = Lock()


    def sample_point(self):
        with self.lock:
            if self.index > self.actual_started_index:
                with open(LAST_IDX_LOGFILE, 'w') as f:
                    f.write(str(self.index))
            point = self.points.iloc[self.index]
            sampleid = point.sampleid
            geom = point.geometry
            lon, lat = geom.x, geom.y

            if self.index % 100 == 0:
                elapsed_seconds = (time.time() - start_time)
                eta = elapsed_seconds / (self.index - self.actual_started_index + 1) * (self.end_index - self.start_index)
                print("index:", self.index, 'elapsed_time:', str(timedelta(seconds=elapsed_seconds)), 'estimated_remaining_time:',  str(timedelta(seconds=eta-elapsed_seconds)))
            self.index += 1
            return [lon, lat], sampleid



def get_collection(sensor, processing_level):

    if sensor == 'S2' and processing_level == 'TOA':
        coll = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    elif sensor == 'L8' and processing_level == 'SR':
        coll = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    else:
        NotImplementedError("Only S2 (TOA) and L8 (SR) implemented at the moment")
        
    return coll


def filter_collection(collection, aoi, period):

    MAX_IMAGES = 15
    
    filtered = collection.filterDate(*period).filterBounds(aoi)  # filter time and region
    if filtered.size().getInfo() == 0:
        print(f'ImageCollection.filter: No suitable images found between {period[0]} and {period[1]}.')
        return None        

    if sensor == 'S2':
        cloud_attr = 'CLOUDY_PIXEL_PERCENTAGE'
    else:
        cloud_attr = 'CLOUD_COVER'


    containsFilter = ee.Filter.contains( leftField= '.geo',rightValue=aoi.bounds())
    filtered = filtered.filter(containsFilter)

    if filtered.size().getInfo() == 0:
        print(f'ImageCollection.filter: No images that fully cover AOI found between {period[0]} and {period[1]}.')
        return None  

    filtered = filtered.limit(MAX_IMAGES, cloud_attr).map(lambda img: img.clip(aoi))

    return filtered


def adjust_coords(coords, old_size, new_size):
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [coords[0][0] + ((xoff + new_size[1]) * xres), coords[0][1] - ((yoff + new_size[0]) * yres)]
    ]


def get_properties(image):
    properties = {}
    for property in image.propertyNames().getInfo():
        properties[property] = image.get(property)
    return ee.Dictionary(properties).getInfo()


def get_patch(image, region, sensor, bands):
    patch = image.select(*bands)

    if sensor == 'S2':
        scale = 10
    elif sensor in ['L7', 'L8']:
        scale = 30
    patch = image.select(*bands).clipToBoundsAndScale(region, scale=scale).sampleRectangle(region, defaultValue=0)

    features = patch.getInfo()  # the actual download
    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(features['properties'][band])        
        raster[band] = img

    coords = np.array(features['geometry']['coordinates'][0])
    coords = [
        [coords[:, 0].min(), coords[:, 1].max()],
        [coords[:, 0].max(), coords[:, 1].min()]
    ]

    quality_band = get_quality_band(sensor)

    pixel_count_ratio = 36 if sensor == 'S2' else 1 # Ratio of quality band pixels to other bands pixel count

    opaque_pixel_ratio = int(100*int(image.get('opaque_clouds').getInfo()[quality_band])/(raster[band].size/pixel_count_ratio))
    cirrus_pixel_ratio = int(100*int(image.get('cirrus_clouds').getInfo()[quality_band])/(raster[band].size/pixel_count_ratio))

    cloud_perc_property_name = 'CLOUDY_PIXEL_PERCENTAGE' if sensor == 'S2' else 'CLOUD_COVER_LAND'


    patch = OrderedDict({
        'raster': raster,
        'coords': coords,
        'metadata': {
            'image_id': image.id().getInfo(),
            'opaque_clouds': opaque_pixel_ratio,
            'cirrus_clouds': cirrus_pixel_ratio,
            'CLOUD_PERC': round(image.getInfo()['properties'][cloud_perc_property_name], 2)
        }
    })
    return patch


def get_cloud_pixel_count_S2(image, pixel_value):

    image = image.unmask()
    cloud = image.updateMask(image).eq(pixel_value)
    
    cloud_total = cloud.reduceRegion(
        reducer=ee.Reducer.sum(),
        scale=60)

    return cloud_total



def get_cloud_pixel_count_L8(qa_band, bit_mask):

    def _mask(qa, bit_mask):
        cloudmask = (1 << bit_mask)
        mask = qa.bitwiseAnd(cloudmask).eq(0)
        return qa.updateMask(mask)

    masked_img = _mask(qa_band, bit_mask)
    mask= masked_img.mask().Not().selfMask()
    
    #use a reducer for counting the masked pixels

    cloud_total = mask.reduceRegion(
        reducer=ee.Reducer.count(),
        scale=60)


    return cloud_total


def set_cloudness_property(image, quality_band):

    if sensor == 'S2':
        cloud_value = 1024
        cirrus_value = 2048
        opaque_cloud_pixels = get_cloud_pixel_count_S2(image.select(quality_band), cloud_value)
        cirrus_cloud_pixels = get_cloud_pixel_count_S2(image.select(quality_band), cirrus_value)


    elif sensor == "L8":
        opaque_cloud_pixels = get_cloud_pixel_count_L8(image.select(quality_band), 3) #Cloud cover is bit 3
        cirrus_cloud_pixels = get_cloud_pixel_count_L8(image.select(quality_band), 2) #Cirrus cover is bit 2


    image_with_cloudness = image.set(
        {'opaque_clouds': opaque_cloud_pixels,
        'cirrus_clouds' : cirrus_cloud_pixels
        }
        )

    return ee.Image(image_with_cloudness)


def get_least_cloudy_patch(period, collection, coords, radius, sensor, bands):

    aoi = ee.Geometry.Point(coords).buffer(radius).bounds()

    limited_collection = filter_collection(collection, aoi, period)
    if limited_collection is None:
        return None

    #Find best image from actually looking at the QA band values in the area of interest
    least_cloudy_image = get_least_cloudy_image_from_coll(limited_collection, sensor)

    least_cloudy_patch = get_patch(least_cloudy_image, aoi, sensor, bands)

    return least_cloudy_patch



def get_quality_band(sensor):
    if sensor == "S2":
        quality_band = 'QA60'
    elif sensor == "L8":
        quality_band = 'QA_PIXEL'
    else:
        raise NotImplementedError("Only S2 and L8 currently allowed")
    
    return quality_band


def get_least_cloudy_image_from_coll(collection, sensor):
    """Generate a list of periods (each period is a tuple of dates).

    Args:
        collection (ee.ImageCollection): contains quality_band band

    Returns:
        ee.Image: least cloudy image based 
    """

    quality_band = get_quality_band(sensor)

    collection = collection.map(lambda img: set_cloudness_property(img, quality_band))

    #Get image(s) with least pixels classified as opaque clouds
    opaque_clouds_pixel_counts = collection.aggregate_array('opaque_clouds').getInfo()
    opaque_clouds_pixel_counts = np.array([item[quality_band] for item in opaque_clouds_pixel_counts])
    least_opaque_clouds_indices = np.where(opaque_clouds_pixel_counts == opaque_clouds_pixel_counts.min())[0]
    
    #If there is a clear winner, use that one
    if len(least_opaque_clouds_indices) == 1:
        least_cloudy_image_idx = least_opaque_clouds_indices[0]
    #Else, among the top images wrt opaque clouds, find the one with least cirrus clouds
    else:
        cirrus_pixel_counts = collection.aggregate_array('cirrus_clouds').getInfo()
        cirrus_pixel_counts = np.array([item[quality_band] for item in cirrus_pixel_counts])
        least_cloudy_candidates = cirrus_pixel_counts[least_opaque_clouds_indices]
        least_cloudy_image_idx = least_opaque_clouds_indices[np.argmin(least_cloudy_candidates)]

    coll_as_list = collection.toList(collection.size())
    top_boy = ee.Image(coll_as_list.get(int(least_cloudy_image_idx)))
    return top_boy


def get_patches_for_location(collection, coords, periods, buffer, sensor, bands):
    
    patches = [get_least_cloudy_patch(period, collection, coords, buffer, sensor, bands) for period in periods]

    return [patch for patch in patches if patch is not None]


def date2str(date):
    return date.strftime('%Y-%m-%d')


def save_geotiff(img, coords, filename):
   
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(coords[0][0] - xres / 2, coords[0][1] + yres / 2) * Affine.scale(xres, -yres)
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': channels,
        'crs': '+proj=latlong',
        'transform': transform,
        'dtype': img.dtype,
        'compress': 'lzw'
    }
    with rasterio.open(filename, 'w', **profile) as f:
        f.write(img.transpose(2, 0, 1))


def save_patch(raster, coords, path, filename):

    img = np.concatenate([v for k, v in raster.items()],axis=2)
    img = img.astype('uint16')

    save_geotiff(img, coords, os.path.join(path, filename))


def get_time_periods(start_date, end_date, period):
    """Generate a list of periods (each period is a tuple of dates).

    Args:
        start_date (str): in the form of YYYY-MM-DD
        end_date (str):  in the form of YYYY-MM-DD
        period (str): length of one period with units, e.g. 1M, 2W etc.

    Returns:
        list: list of tuples (start_date, end_date) for each period
    """
    start = date(*[int(i) for i in start_date.split('-')]) 
    end = date(*[int(i) for i in end_date.split('-')])
    
    period_length, unit = int(period[:-1]), period[-1]
    assert unit in ["M", "W"], f"Unexpected time unit; expected M for months of W for weeks, got '{unit}'"

    periods = []

    while start < end:
        if unit == 'M':
            periods.append((date2str(start), date2str(start+ relativedelta(months=+period_length))))
        else:
            periods.append((date2str(start), date2str(start+ relativedelta(weeks=+period_length))))
        if unit == 'M':
            start += relativedelta(months=+period_length)
        else:
            start += relativedelta(weeks=+period_length)

    return periods


def resample(src, width, height):

    # resample data to target shape
    data = src.read(
        out_shape=(
            src.count,
            height,
            width
        ),

        resampling=Resampling.bilinear
    )

    # scale image transform
    out_transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )

    out_profile = src.profile
    out_profile.update(
        {
            'width': width,
            'height': height,
            'transform': out_transform
            }
        )

    return data, out_profile



class Counter:

    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()

    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--sample_points_path', type=str,default="../data/brasil_coverage_2018_sample.csv")
   
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_freq', type=int, default=100)

    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--save_path', type=str,default="../data/images-2018")

    parser.add_argument('--sensor', type=str, default='S2')
    parser.add_argument('--bands', type=str, default='S2_RGBNIR')
    parser.add_argument('--processing_level', type=str, default='TOA') 
    
    parser.add_argument('--buffer', type=int, default=750)

    parser.add_argument('--start_date', type=str, default='2018-01-01')
    parser.add_argument('--end_date', type=str, default='2019-01-01')
    parser.add_argument('--period_length', type=str, default='12M')
    parser.add_argument('--target_shape', type=tuple, default=(256,256))
    parser.add_argument('--logs_file', type=str, default='logs.txt')
    

    args = parser.parse_args()
    
    ee.Initialize()

    #y, m, d, h, m, s, _  = datetime.datetime.now()
    #with open(args.logs_file, "a") as source:
    #    source.write(f"Run started at {y}-{m}-{d}-{h}-{m}-{s}")


    sensor = args.sensor
    buffer = args.buffer    
    collection = get_collection(sensor=sensor, processing_level=args.processing_level)

    sampler = ShapeFileSampler(args.sample_points_path)
    start_index = sampler.index
    end_index = sampler.points.index[-1] + 1

    bands = BANDS[args.bands]

    time_periods = get_time_periods(args.start_date, args.end_date, args.period_length)
    start_time = time.time()
    counter = Counter()
    target_shape = args.target_shape


    log_file_lock = Lock()

    def worker(idx):
        try:
            coords, sampleid = sampler.sample_point()
            # sampleid = int(sampleid)
            # print(sampleid,type(sampleid))
            out_folder = os.path.join(args.save_path, f'{sampleid}')

            if (os.path.isdir(out_folder) and  os.listdir(out_folder)):
                print(f'Folder for sampleid {sampleid} already contains data; skipping')
                return

            os.makedirs(out_folder, exist_ok=True)

            patches = get_patches_for_location(collection, coords, time_periods, buffer, sensor, bands)

            for patch in patches:
                out_name = f"{patch['metadata']['image_id']}_opaque_clouds_{patch['metadata']['opaque_clouds']}_cirrus_{patch['metadata']['cirrus_clouds']}_CLOUD_PERC_{patch['metadata']['CLOUD_PERC']}.tif"
                save_patch(
                    raster=patch['raster'],
                    coords=patch['coords'],
                    path=out_folder,
                    filename=out_name,
                )
                shape = patch['raster'][next(iter(patch['raster']))].shape
                if (shape[0], shape[1]) != target_shape:
                    with rasterio.open(os.path.join(out_folder, out_name)) as src:
                        out_image, out_profile = resample(src, target_shape[0], target_shape[1])
                    with rasterio.open(os.path.join(out_folder, out_name), "w", **out_profile) as sink:
                        sink.write(out_image)

            assert os.path.isdir(out_folder), f'folder not created: sampleid {sampleid}, idx {idx}'

            count = counter.update(len(patches))
            if count % args.log_freq == 0:
                print(f'Downloaded {count} images in {time.time() - start_time:.3f}s.')

        except Exception as e:
            print(f'Failed for sampleid {sampleid} with error: {e}')
            with log_file_lock:
                with open(args.logs_file, "a") as source:
                    source.write(f'Failed for sampleid {sampleid} with error: {e}')

    indices = range(start_index, end_index)
    print("Starting")
    if args.num_workers == 0:
        for i in indices:
            worker(i)
            exit()
    else:
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
