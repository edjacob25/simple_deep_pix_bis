import cv2 as cv
import math
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from random import sample
from typing import List, Tuple


def get_coordinates_of_eyes_in_frame(file_path: Path, frame: int) -> List[int]:
    whichever = False
    with file_path.open("r") as f:
        last = None
        for line in f:
            _, eyes = line.split(",", maxsplit=1)
            eyes = [int(x) for x in eyes.split(",")]
            all_zeros = all([x == 0 for x in eyes])
            if not all_zeros:
                last = eyes
            if line.startswith(str(frame)) or whichever:
                if all_zeros:
                    whichever = True
                    continue
                return eyes
        return last


def get_slices(eyes_points: List[int]) -> Tuple[Tuple, Tuple]:
    x_eye_left, y_eye_left, x_eye_right, y_eye_right = eyes_points

    x_side_distance = math.floor((224 - math.fabs(x_eye_left - x_eye_right)) / 2)
    y_side_distance = math.floor((224 - math.fabs(y_eye_left - y_eye_right)) / 2)
    min_x = min(x_eye_left, x_eye_right) - x_side_distance
    max_x = max(x_eye_left, x_eye_right) + x_side_distance

    min_y = min(y_eye_left, y_eye_right) - y_side_distance
    max_y = max(y_eye_left, y_eye_right) + y_side_distance

    if max_x - min_x == 223:
        min_x = min_x - 1

    if max_y - min_y == 223:
        min_y = min_y - 1

    return (min_x, max_x), (min_y, max_y)


def save_frame(fname: Path, save_dir: Path, frame_idx=1):
    data_file = fname.with_suffix(".txt")
    eyes = get_coordinates_of_eyes_in_frame(data_file, frame_idx)
    xs, ys = get_slices(eyes)

    v = cv.VideoCapture(str(fname))
    v.set(cv.CAP_PROP_POS_FRAMES, frame_idx - 1)

    ret, frame = v.read()
    v.release()
    frame = frame[xs[0]: xs[1], ys[0]: ys[1], :]
    frame = np.moveaxis(frame, -1, 0)

    if frame.shape != (3, 224, 224):
        resized = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)
        frame = resized

    img = Image.fromarray(frame)

    save_path = save_dir / f"{fname.name}__{frame_idx}.png"

    img.save(save_path)


def convert_directory(path: Path, save_dir: Path, frames_per_video: int = 15):
    for item in path.iterdir():
        if item.suffix == ".avi":
            v = cv.VideoCapture(str(item))
            frames_in_video = int(v.get(cv.CAP_PROP_FRAME_COUNT))
            v.release()
            if frames_per_video > frames_in_video:
                indexes = [x for x in range(frames_in_video)]
            else:
                indexes = sample([x for x in range(frames_in_video)], frames_per_video)
            for i in indexes:
                save_frame(item, save_dir, i)


def main():
    parser = ArgumentParser()
    parser.add_argument("directory", help="Directory of the videos")
    parser.add_argument("save_dir", help="Directory to save")
    parser.add_argument("-nf", "--num_frames", type=int, help="Frames per video to save", default=10)
    args = parser.parse_args()

    data_dir = Path(args.directory)
    save_dir = Path(args.save_dir)

    if not data_dir.exists() or not data_dir.is_dir():
        print("Directory does not exists or is not a directory")
        exit(1)

    if not save_dir.is_dir():
        print("Saving Directory is not a directory")
        exit(1)

    if not save_dir.exists():
        save_dir.mkdir()

    for directory in data_dir.iterdir():
        save_path = save_dir / directory.name
        convert_directory(directory, save_path, args.num_frames)


if __name__ == '__main__':
    main()
