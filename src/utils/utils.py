import platform

import cv2
import pandas as pd


def check_os():
    os = platform.system()

    if os == 'Darwin':
        return "MacOS"
    elif os == 'Linux':
        return "Linux"
    else:
        return "Unknown OS"


def read_annotation(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            elements = line.strip().split(' ')

            id_ = elements[0]
            class_ = elements[1]
            x = int(elements[2])
            y = int(elements[3])
            width = int(elements[4])
            height = int(elements[5])
            occlusion = int(elements[6])

            data.append({
                'id': id_,
                'class': class_,
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'occlusion': occlusion
            })
    return data


def read_image(filename):
    return cv2.imread(filename)


def read_json(filename):
    import json
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def sort_key(filename):
    # Extract the numeric portion of the filename and convert to an integer
    numeric_portion = filename.split('_')[1]
    return int(numeric_portion)


def add_metadata_info(metadata_clip, frame, border_size=200, y_initial=20, y_gap=20):
    frame = cv2.copyMakeBorder(frame, 0, border_size, 0, 0, cv2.BORDER_CONSTANT, None, value=0)
    original_height, _, _ = frame.shape
    y_initial = original_height + y_initial
    # Extract metadata for the current clip
    folder_name = metadata_clip['Folder name'].values[0]
    datetime = metadata_clip['DateTime'].values[0]
    temperature = metadata_clip['Temperature'].values[0]
    humidity = metadata_clip['Humidity'].values[0]
    precipitation = metadata_clip['Precipitation latest 10 min'].values[0]
    dew_point = metadata_clip['Dew Point'].values[0]
    wind_direction = metadata_clip['Wind Direction'].values[0]
    wind_speed = metadata_clip['Wind Speed'].values[0]
    sun_radiation = metadata_clip['Sun Radiation Intensity'].values[0]
    sunshine_min = metadata_clip['Min of sunshine latest 10 min'].values[0]

    # During your frame rendering loop, add the metadata information to the frame:
    cv2.putText(frame, f"Folder: {folder_name}", (5, y_initial), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"DateTime: {datetime}", (5, y_initial + y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"Temperature: {temperature}", (5, y_initial + 2 * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Humidity: {humidity}", (5, y_initial + 3 * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200),
                1, cv2.LINE_AA)
    cv2.putText(frame, f"Precipitation: {precipitation}", (5, y_initial + 4 * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Dew Point: {dew_point}", (5, y_initial + 5 * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Wind Direction: {wind_direction}", (5, y_initial + 6 * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Wind Speed: {wind_speed}", (5, y_initial + 7 * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Sun Radiation: {sun_radiation}", (5, y_initial + 8 * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Sunshine Min: {sunshine_min}", (5, y_initial + 9 * y_gap), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200), 1, cv2.LINE_AA)


def calculate_f1_score(predictions_df, gt_df):
    # Merge the dataframes on 'id'
    merged_df = pd.merge(predictions_df, gt_df, on='id')
    true_positive = sum((merged_df['loitering_predicted'] == 1) & (merged_df['loitering'] == 1))
    false_positive = sum((merged_df['loitering_predicted'] == 1) & (merged_df['loitering'] == 0))
    false_negative = sum((merged_df['loitering_predicted'] == 0) & (merged_df['loitering'] == 1))
    true_negative = sum((merged_df['loitering_predicted'] == 0) & (merged_df['loitering'] == 0))

    # calculate and print accuracy, precision, recall, and F1-score, sensitivity, specificity
    total = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative) / total if total != 0 else 0

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive != 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    sensitivity = true_positive / (true_positive + false_negative) if true_positive + false_negative != 0 else 0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive != 0 else 1
    roc_auc = (sensitivity + specificity) / 2

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"ROC AUC: {roc_auc}")

    return f1


if __name__ == "__main__":
    # path = r'/Users/johnny/Projects/datasets/harbor-synthetic/annotations/20200514/clip_0_1331/annotations_0000.txt'
    # lines = read_annotation(path)
    path = r'/Users/johnny/Projects/datasets/harbor-synthetic/LTD_Dataset/LTD_Dataset/Image_Dataset_25fps/20200514/clip_0_1331/image_0000.jpg'
    read_image(path)
