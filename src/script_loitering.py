import os
import pickle
import platform
import shutil

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.loitering_methods import (ellipse_loitering_detection,
                                   find_loitering_start_and_end,
                                   RectangleLoiteringDetection,
                                   closed_areas_loitering_detection,
                                   sector_loitering_detection,
                                   no_motion_loitering_detection,
                                   convex_hull_loitering_detection)


def check_os():
    os = platform.system()
    if os == "Darwin":
        return "MacOS"
    elif os == "Linux":
        return "Linux"
    else:
        return "Unknown OS"


operating_system = check_os()
root_path = "/Users/johnny/Projects/" if operating_system == "MacOS" else "/home/johnny/Projects/"

directories = [
    "plots/trajectory",
    "plots/rectangle",
    "plots/ellipse",
    "plots/sector",
    "plots/closed_areas",
    "plots/no_motion",
    "plots/second_derivative",
    "plots/convex_hull"
]

# Delete and recreate each directory
for directory in directories:
    shutil.rmtree(directory, ignore_errors=True)  # Delete the directory if it exists
    os.makedirs(directory, exist_ok=True)  # Create the directory

# read pickle files
with open(root_path + "/Master_Thesis_CV/datasets/test_dict_smooth_ipm.pkl", "rb") as f:
    coord_dict_smooth = pickle.load(f)

# Create a DataFrame to store the results
df_predictions = pd.DataFrame()
df_predictions['id'] = coord_dict_smooth.keys()
for method in ["rectangle", "convex_hull", "ellipse", "closed_areas", "sector", "no_motion_short_term",
               "no_motion_long_term"]:
    df_predictions[method] = 0  # Initialize with 0 (no loitering)

# Parameters
"""theta = 60
feature_points_threshold = 5
window_size = 21
S_threshold_Rectangle = 0.14938462723902313
T0_Rectangle = 49
S_threshold_convex = 0.14938462723902313
S_threshold_Ellipse = 0.05
S_threshold_closed = 0.14938462723902313
T0_sector = 49
M1 = 9
M2 = 38
S_threshold_sector = 100
std = 20.69563455729097
radius = 53
frame_threshold = 3
mode_no_motion = 'short_term'
display = True"""

#GLOBAL

theta = 13
feature_points_threshold = 4
window_size = 4

# ELLIPSE METHOD
S_threshold_Ellipse = 72.26410079216053

# CONVEX HULL METHOD
S_threshold_convex = 59.70129415254876

# RECTANGLE METHOD
S_threshold_Rectangle = 196.75825780847265
T0_Rectangle = 2

# CLOSED AREAS METHOD
S_threshold_closed = 0.04452023079933021

# NO MOTION METHOD SHORT
std_short = 22.162147086301772
radius_short = 60
frame_threshold_short = 7

# NO MOTION LONG
std = 22.162147086301772
radius = 60
frame_threshold = 94

# SECTOR METHOD
T0_sector = 51
M1 = 60
M2 = 112
S_threshold_sector = 41.81589612663217

display = True
"""
# Updated variables
theta = 16
feature_points_threshold = 5
window_size = 5

# ELLIPSE METHOD
S_threshold_Ellipse = 168.32963073666767

# CONVEX HULL METHOD
S_threshold_convex = 10.091478766795216
# Note: I'm not sure where S_threshold_convex2 should be placed. I'm adding it below just in case.
S_threshold_convex2 = 195.93219848499774

# RECTANGLE METHOD
S_threshold_Rectangle = 83.56839043422302
T0_Rectangle = 12

# CLOSED AREAS METHOD
S_threshold_closed = 0.30508943840043223

# NO MOTION METHOD SHORT
# Note: I don't see new values for the short method in the provided list, so I'm leaving them as they are.
std_short = 22.162147086301772
radius_short = 60
frame_threshold_short = 7

# NO MOTION LONG
std = 55.0666534039824
radius = 45
frame_threshold = 90

# SECTOR METHOD
T0_sector = 40
M1 = 21
M2 = 6
S_threshold_sector = 173.83740033832524
display = False
"""

# Processing loop
for i in tqdm(range(len(coord_dict_smooth))):
    ID, coordinates = list(coord_dict_smooth.items())[i]

    coords_array = np.array(coordinates)

    feature_point_start, feature_point_end = find_loitering_start_and_end(coords_array, theta, window_size,
                                                                          feature_points_threshold)
    
    # plot trajectory
    if display:
        plt.figure()
        plt.plot(coords_array[:, 0], coords_array[:, 1], "b")
        plt.title(f"Trajectory: {ID}")
        plt.gca().invert_yaxis()
        plt.xlim([0, 384])
        plt.ylim([288, 0])
        plt.gca().set_xticks([])  # Remove x-axis tick marks
        plt.gca().set_yticks([])  # Remove y-axis tick marks
        plt.savefig(f"plots/trajectory/{ID}.pdf")
        plt.close()

    if feature_point_start is not None:
        if ellipse_loitering_detection(coords_array, feature_point_start, feature_point_end, S_threshold_Ellipse, ID, display=display):
            df_predictions.loc[df_predictions["id"] == ID, "ellipse"] = 1

        if convex_hull_loitering_detection(coords_array, feature_point_start, feature_point_end, S_threshold_convex,
                                           ID, display=display):
            df_predictions.loc[df_predictions["id"] == ID, "convex_hull"] = 1

        loitering_detector = RectangleLoiteringDetection(coords_array, theta, T0_Rectangle, S_threshold_Rectangle)
        if loitering_detector.detect_loitering():
            df_predictions.loc[df_predictions["id"] == ID, "rectangle"] = 1
            if display:
                loitering_detector.plot_trajectory(ID)

        if closed_areas_loitering_detection(coords_array, S_threshold_closed, ID, display=display):
            df_predictions.loc[df_predictions["id"] == ID, "closed_areas"] = 1

        if sector_loitering_detection(coords_array, T0_sector, M1, M2, S_threshold_sector, ID, display=display):
            df_predictions.loc[df_predictions["id"] == ID, "sector"] = 1

    else:
        if no_motion_loitering_detection(coords_array, mode="short_term", frame_threshold=frame_threshold_short,
                                         radius=radius_short, std_threshold=std_short, ID=ID, display=display):
            df_predictions.loc[df_predictions["id"] == ID, "no_motion_short_term"] = 1

        if no_motion_loitering_detection(coords_array, mode="long_term", frame_threshold=frame_threshold,
                                         radius=radius, std_threshold=std, ID=ID, display=display):
            df_predictions.loc[df_predictions["id"] == ID, "no_motion_long_term"] = 1

# Save the DataFrame
df_predictions.to_csv(root_path + "/Master_Thesis_CV/datasets/predictions_NO_IMP.csv", index=False)
