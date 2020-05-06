from typing import List, Tuple
import math
from pathlib import Path


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

    x_side_distance = math.floor((448 - math.fabs(x_eye_left - x_eye_right)) / 2)
    y_side_distance = math.floor((448 - math.fabs(y_eye_left - y_eye_right)) / 2)
    min_x = min(x_eye_left, x_eye_right) - x_side_distance
    max_x = max(x_eye_left, x_eye_right) + x_side_distance

    min_y = min(y_eye_left, y_eye_right) - y_side_distance
    max_y = max(y_eye_left, y_eye_right) + y_side_distance

    if max_x - min_x == 448:
        min_x = min_x - 1

    if max_y - min_y == 448:
        min_y = min_y - 1

    return (min_x, max_x), (min_y, max_y)
