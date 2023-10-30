import os
import time

import cv2
import imageio
import pandas as pd
import ujson

from src.utils.utils import check_os, read_json, read_annotation, read_image


def rendering_video(cfg, output_path, slow_factor=1.0):
    operating_system = check_os()

    if operating_system == "MacOS":
        root_path = "/Users/johnny/Projects/"
    elif operating_system == "Linux":
        root_path = "/home/johnny/Projects/"

    # ------------------- #
    from tqdm import tqdm

    # read the json annotations
    annotations = read_json(cfg.annotations_path)
    ids_list = []

    # read the ids from the csv
    try:
        ids_df = pd.read_csv(root_path + "Master_Thesis_CV/datasets/ids.csv")
    except:
        ids_df = None

    if ids_df is not None:
        ids_list = ids_df['id'].tolist()
        ids_list = [int(x) for x in ids_list]
        # Make a set of unique folder and clip combinations from the CSV for faster lookup
        folder_clip_set = set(
            (str(folder), str(clip)) for folder, clip in ids_df[['folder', 'clip']].drop_duplicates().values.tolist())

        # Filter the JSON based on the folder-clip combinations from the CSV
        filtered_annotations = {}
        for folder in tqdm(annotations, desc="Filtering annotations", position=0, leave=False):
            if folder in [x[0] for x in folder_clip_set]:  # Check if folder is in our set
                filtered_annotations[folder] = {}
                for clip in annotations[folder]:
                    if (folder, clip) in folder_clip_set:  # Check if the folder-clip combination is in our set
                        filtered_annotations[folder][clip] = annotations[folder][clip]

        annotations = filtered_annotations

    if cfg.folder in annotations:
        filtered_annotations = {cfg.folder: annotations[cfg.folder]}
    else:
        print(f"Folder {cfg.folder} not found in annotations!")
        return

    annotations = filtered_annotations

    # ------------------- #
    # Order the annotations by ID ascendent
    # ------------------- #
    wait_time = 1
    if cfg.metadata is not None:
        metadata = pd.read_csv(root_path + cfg.metadata)
    time_start = time.time()
    # filter annotations by parameter
    cfg.folder
    for date in tqdm(annotations):
        if True:
            for clip in tqdm(annotations[date]):
                if clip == cfg.clip_reproduce or cfg.clip_reproduce == "" or cfg.clip_reproduce is None:
                    metadata_clip = metadata[metadata['Clip Name'] == clip]
                    # print(metadata_clip)
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    if not os.path.exists(output_path + "/" f"{date}"):
                        os.makedirs(output_path + "/" f"{date}", exist_ok=False)
                    out = cv2.VideoWriter(output_path + "/" f"{date}/" + f"{date}_{clip}_video.mp4", fourcc, 1,
                                          (384, 288))
                    object_centroids = {}
                    list_frames = []
                    list_position = []
                    print(f"Rendering {date}_{clip}_video.mp4")
                    print("-------------------")
                    for frame_num in annotations[date][clip]:
                        print("-------------------")
                        path_frame = annotations[date][clip][frame_num]['path_frame']
                        path_annotation = annotations[date][clip][frame_num]['path_annotation']

                        frame = read_image(path_frame)
                        annotation = read_annotation(path_annotation)
                        for obj in annotation:
                            if obj['class'] == 'human':
                                id_aux = obj['id'].lstrip('0')
                                id_aux = int(id_aux)
                                print("ID: ", id_aux)
                                if id_aux in ids_list:
                                    frame = cv2.rectangle(frame, (obj['x'], obj['y']),
                                                          (obj['width'], obj['height']), (0, 0, 255), 1)
                                    obj['id'] = obj['id'].lstrip('0')

                                    # compare if obj['x] and obj['y'] are in the list
                                    frame = cv2.putText(frame, obj['id'], (obj['x'] - 3, obj['y'] - 3),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                                    centroid_x = int((obj['x'] + obj['width']) / 2)
                                    centroid_y = int((obj['y'] + obj['height']) / 2)
                                    centroid = (centroid_x, centroid_y)

                                    if obj['id'] not in object_centroids:
                                        object_centroids[obj['id']] = []
                                    object_centroids[obj['id']].append(centroid)

                                    for i in range(1, len(object_centroids[obj['id']])):
                                        start = object_centroids[obj['id']][i - 1]
                                        end = object_centroids[obj['id']][i]
                                        frame = cv2.line(frame, start, end, (0, 0, 255), 1)
                                else:
                                    frame = cv2.rectangle(frame, (obj['x'], obj['y']),
                                                          (obj['width'], obj['height']), (0, 255, 0), 1)
                                    obj['id'] = obj['id'].lstrip('0')
                                    # compare if [obj['x'].obj['y']] are in the list
                                    if [obj['x'], obj['y']] in list_position:
                                        frame = cv2.putText(frame, obj['id'], (obj['x'] - 10, obj['y'] - 10),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.25,
                                                            (0, 255, 0), 1)

                                    else:
                                        frame = cv2.putText(frame, obj['id'], (obj['x'] - 3, obj['y'] - 3),
                                                            cv2.FONT_HERSHEY_SIMPLEX,
                                                            0.25,
                                                            (0, 255, 0), 1)

                                    centroid_x = int((obj['x'] + obj['width']) / 2)
                                    centroid_y = int((obj['y'] + obj['height']) / 2)
                                    centroid = (centroid_x, centroid_y)

                                    if obj['id'] not in object_centroids:
                                        object_centroids[obj['id']] = []
                                    object_centroids[obj['id']].append(centroid)

                                    for i in range(1, len(object_centroids[obj['id']])):
                                        start = object_centroids[obj['id']][i - 1]
                                        end = object_centroids[obj['id']][i]
                                        frame = cv2.line(frame, start, end, (0, 255, 0), 1)
                            list_position.append([obj['x'], obj['y']])

                        cv2.putText(frame, f"Frame: {frame_num}", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200),
                                    1,
                                    cv2.LINE_AA)
                        out.write(frame)
                        list_frames.append(frame)
                        if cfg.display:
                            cv2.imshow(f"{date}_{clip}_video", frame)
                            k = cv2.waitKey(int(wait_time * slow_factor))  # dd& 0xFF
                            if k == ord('q'):  # stop playing
                                break
                            elif k == ord('s'):  # save current frame
                                cv2.imwrite(f'save_{frame_num}.png', frame)
                            elif k == ord('p'):  # pause the video
                                cv2.waitKey(-1)  # wait until any key is pressed
                            elif k == ord('r'):  # resume the video
                                continue
                            elif k == ord('b'):
                                break
                            elif k == ord('d'):
                                slow_factor = slow_factor - 1
                                print(slow_factor)
                            elif k == ord('i'):
                                slow_factor = slow_factor + 1
                                print(slow_factor)

                if cfg.tracking:
                    with open(output_path + "/" f"{date}/" + f"{clip}.json", 'w') as f:
                        ujson.dump(object_centroids, f)
                if cfg.save:
                    # Close the video writer
                    out.release()
                    # Close the player window
                cv2.destroyAllWindows()

                if cfg.gif:
                    # os.system(f"ffmpeg -i {output_path}/{date}/{date}_{clip}_video.mp4 -vf 'fps=1,scale=384:288' {output_path}/{date}/{date}_{clip}_video.gif")
                    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in list_frames]
                    # Guardar el GIF utilizando imageio
                    imageio.mimsave(output_path + "/" f"{date}/" + f"{date}_{clip}_video.gif", rgb_frames, duration=1)
    time_end = time.time()
    print(f"Total Time: {time_end - time_start}")
