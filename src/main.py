import argparse
import os

from src.rendering import rendering_video

TOTAL_FRAMES_VIDEO = 120


def main(cfg):
    current_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_path, "results", f'{cfg.run_name}')
    os.makedirs(output_path, exist_ok=True)
    print(
        f"Run Name: {cfg.run_name} - Output Path: {output_path}")
    print("----------------------------------------")
    rendering_video(cfg, output_path, slow_factor=cfg.slow_factor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run_name", required=True, type=str, help="Run Folder Name")
    parser.add_argument("-f", "--folder", required=True, type=str, help="Folder Date")
    parser.add_argument("-a", "--annotations_path", required=True, type=str, help="Path to the annotations")
    parser.add_argument("-s", "--save", default=True, type=bool, help="Save the video or not")
    parser.add_argument("-t", "--tracking", default=False, type=bool, help="Save annotations")
    # parser.add_argument("-c", "--clip_reproduce", type=str, default="clip_0_1331", help="Clip to reproduce")
    parser.add_argument("-c", "--clip_reproduce", type=str, default="", help="Clip to reproduce")
    parser.add_argument("-d", "--display", default=True, type=bool, help="Show the video or not")
    parser.add_argument("-m", "--metadata", required=False,
                        default='datasets/harbor-synthetic/LTD_Dataset/LTD_Dataset/metadata.csv', type=str,
                        help="Path to the annotations")
    parser.add_argument("-sf", "--slow_factor", default=144, type=float, help="Slow factor")
    parser.add_argument("-g", "--gif", default=True, type=bool, help="Save gif or not")

    args = parser.parse_args()
    print("----------------------------------------")
    print("Finished reading config file")

    main(args)
