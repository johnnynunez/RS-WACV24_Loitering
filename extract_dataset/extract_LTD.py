# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:33:16 2021

@author: Ivan
modified by Jacob
"""

import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import platform

# import torch
# import torch.utils.data

from load_video_metadata import VideoClipLoader


def check_os():
    os = platform.system()
    if os == "Darwin":
        return "MacOS"
    elif os == "Linux":
        return "Linux"
    else:
        return "Unknown OS"

operating_system = check_os()
root_path = "/Users/jacob/Projects/" if operating_system == "MacOS" else "/home/jacob/Projects/"

'''
METADATA structure:

    Folder name - name of the day folder in the format of YYYYMMDD
    Clip Name - name of the clip folderin the format clip_{number}_{HHMM}
    DateTime - the date time when the 2 min has started
    Temperature - in C
    Humidity - in %
    Precipitation - in kg/m2
    Dew Point - in C
    Wind Direction - in degrees
    Wind Speed - in m/s2
    Sun Radiation Intensity - W/m2
    Min of sunshine latest 10 min - in minutes

'''

'''
Config file structure:

    video_path - top path that contains all day video folders and the metadata csv
    metadata - the selected metadata dataframe that will be used to select the clips
    main_save_path - the top path for saving the created images and metadata
    every - sampling rate at which the images will be created from the video, default one is every 25th
    pattern_video - what pattern to be searched for in the folders, default one is .mp4

'''




# Script to get the queried metadata and create image set for it
def make_imagesets(cfg, csv_vids=None):

    # get the dataset from the selected metadata and the specified "every" next one
    dataset = VideoClipLoader(video_dir=cfg['video_path'], metadata = cfg['metadata'], every = cfg['every'])


    # create the folder
    if not os.path.exists(cfg['main_save_path']):

        os.makedirs(cfg['main_save_path'])
    #  go through all the clips in the dataset, cut them in images and save them
    metadata_list = []

    if csv_vids:
        df = pd.read_csv(csv_vids, index_col='index')

    for i in tqdm(range(len(dataset))):
        if csv_vids:
            if i not in df.index:
                continue

        img, path, curr_metadata = dataset[i]

        curr_metadata = curr_metadata.split(',')
        #  save dir for all the images from each clip
        save_dir = os.path.join(cfg['main_save_path'],curr_metadata[0],curr_metadata[1])

        #  create the save dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            continue

        #  go through all the images from the clip, transform them and save them to disk
        count = 0
        for sub_img in img:
            sub_img = sub_img.unsqueeze(0)
            input_img = sub_img.squeeze(1).mul(255).byte().numpy()

            count_str ="image_" + str(count).zfill(4)
            cv2.imwrite(os.path.join(save_dir, count_str + '.jpg'), np.squeeze(input_img))
            #  create a metadata entry for the image, containing the image number
            img_metadata = [curr_metadata[0], curr_metadata[1], count_str, curr_metadata[2], curr_metadata[3], curr_metadata[4], curr_metadata[5], curr_metadata[6],curr_metadata[7], curr_metadata[8],curr_metadata[9],curr_metadata[10]]
            metadata_list.append(img_metadata)


            count+=1

        # print(f"Finished {save_dir}")

    # save all the new metadata entries per image to a new csv in the same folder
    metadata_df = pd.DataFrame.from_records(metadata_list, columns=['Folder name', 'Clip Name', 'Image Number', 'DateTime', 'Temperature', 'Humidity', 'Precipitation', 'Dew Point', 'Wind Direction', 'Wind Speed', 'Sun Radiation Intensity', 'Min of sunshine latest 10 min'])
    metadata_df.to_csv(os.path.join(cfg['main_save_path'], "metadata_images.csv"), index=False)




if __name__ == '__main__':

    every = 25
    video_path = root_path + r"/harbor-synthetic/LTD_Dataset/LTD_Dataset"
    main_save_dir = os.path.join(video_path, "Image_Dataset_25fps")

    csv_vids = None
    # csv_vids = r'/home/jacob/code/harbor-synthetic/src/data/split/videos/harbor_train_low_density_0100.csv'
    # csv_vids = r'/home/jacob/code/harbor-synthetic/src/data/split/videos/harbor_test_100_vehicles.csv'


    pattern_vid = ".mp4"

    metadata_name = "metadata.csv"
    metadata_path = os.path.join(video_path, metadata_name)

    metadata = pd.read_csv(metadata_path)
    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = True)




    # Training
    #  creating the cfg file containing all the necessary information for creating the training set
    cfg = {
        # 'video_path': video_path,
        'video_path': os.path.join(video_path, 'Video Clips'),
        'metadata': metadata,
        'main_save_path': main_save_dir,
        'every': every,
        'pattern_video': pattern_vid,

    }
    #  call the function
    make_imagesets(cfg, csv_vids)
