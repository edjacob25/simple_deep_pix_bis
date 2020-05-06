import cv2 as cv

from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
from random import sample
from shutil import rmtree

from nn.common import get_coordinates_of_eyes_in_frame, get_slices

face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def save_frame(fname: Path, save_dir: Path, frame_idx=1):
    v = cv.VideoCapture(str(fname))
    v.set(cv.CAP_PROP_POS_FRAMES, frame_idx - 1)

    ret, frame = v.read()
    v.release()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) < 1:
        print(f"Could not find a face in frame {frame_idx} for file {fname}, using backup method")
        eyes = get_coordinates_of_eyes_in_frame(fname.with_suffix(".txt"), frame_idx)
        xs, ys = get_slices(eyes)
        frame = frame[ys[0]: ys[1], xs[0]: xs[1], :]

    else:
        x, y, w, h = faces[0]
        frame = frame[y: y + h, x: x + h, :]

    if frame.shape != (3, 224, 224):
        try:
            resized = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)
            frame = resized
        except:
            print(f"Error with frame {frame_idx} for file {fname}, has shape {frame.shape}")
            return

    save_path = save_dir / f"{fname.stem}__{frame_idx}.png"
    print(f"Saving frame {frame_idx} to {save_path}")
    cv.imwrite(str(save_path), frame)


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
    parser.add_argument("-d", "--directory", help="Directory of the videos")
    parser.add_argument("-s", "--save_dir", help="Directory to save")
    parser.add_argument("-nf", "--num_frames", type=int, help="Frames per video to save", default=10)
    args = parser.parse_args()

    config = ConfigParser()
    config.read("config.ini")

    data_dir = Path(args.directory) if args.directory else Path(config["ROUTES"]["initial_files"])
    save_dir = Path(args.save_dir) if args.save_dir else (Path(config["ROUTES"]["base_files"]) / "Preprocessed")

    if not data_dir.exists() or not data_dir.is_dir():
        print("Directory does not exists or is not a directory")
        exit(1)

    if save_dir.exists() and not save_dir.is_dir():
        print("Saving Directory is not a directory")
        exit(1)

    if not save_dir.exists():
        save_dir.mkdir()

    for directory in data_dir.iterdir():
        save_path = save_dir / directory.name

        if save_path.exists():
            rmtree(save_path)

        save_path.mkdir()
        print(f"Converting videos from {directory} and saving them to {save_path}, taking {args.num_frames} per video")
        convert_directory(directory, save_path, args.num_frames)


if __name__ == '__main__':
    main()
