#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ageorge
"""

# ==============================================================================
# Import what is needed here:

import math
import random
import traceback
from pathlib import Path
from random import sample
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torch.utils.data as data
from torch import Tensor

random.seed(a=7)


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


def get_frame(fname: Path, frame_idx=1) -> np.ndarray:
    data_file = fname.with_suffix(".txt")
    eyes = get_coordinates_of_eyes_in_frame(data_file, frame_idx)
    xs, ys = get_slices(eyes)

    v = cv.VideoCapture(str(fname))
    v.set(cv.CAP_PROP_POS_FRAMES, frame_idx - 1)

    ret, frame = v.read()
    o_frame_shape = frame.shape
    v.release()
    frame = frame[xs[0]: xs[1], ys[0]: ys[1], :]
    frame = np.moveaxis(frame, -1, 0)

    if frame.shape != (3, 224, 224):
        # print(f"{fname} - {frame.shape} - {o_frame_shape} - {xs} - {ys} - {frame_idx}")
        resized = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)
        frame = resized
    # print(frame.shape)
    return frame


def balance_data_filebased(fnil):
    from collections import Counter

    ''' Return a balanced list of files tuples '''
    reals = [ff for ff in fnil if ff[2] == 1]

    attacks = [ff for ff in fnil if ff[2] == -1]

    num_reals = len(reals)
    num_attacks = len(attacks)

    print("num_reals", len(reals), "num_attacks", len(attacks))

    # Simplest, always downsample, no upsampling here, from each file select a number of frames so as to balance with
    # the other class.

    if num_reals > num_attacks:

        # downsample reals

        names = [ff[0] for ff in reals]

        cnt = Counter(names)

        frames_perfile = round(num_attacks / len(cnt))  # would be low, now add some more to solve the issue

        new_reals = []
        new_attacks = attacks

        for file_name in sorted(cnt.keys()):

            num_files = cnt[file_name]

            try:
                sample_idx = random.sample(range(num_files), frames_perfile)
            except:
                sample_idx = range(num_files)

            ffs = [ff for ff in reals if ff[0] == file_name]

            ffs = [ff for idx, ff in enumerate(ffs) if idx in sample_idx]
            new_reals = new_reals + ffs

        try:
            new_indexes = random.sample(range(len(reals)), len(new_attacks) - len(new_reals))
        except:
            new_indexes = []

        new_reals = new_reals + [ff for idx, ff in enumerate(reals) if idx in new_indexes]

        fnil_balanced = new_reals + new_attacks


    else:
        # downsample attacks        
        names = [ff[0] for ff in attacks]

        cnt = Counter(names)

        frames_perfile = round(num_reals / len(cnt))  # would be low, now add some more to solve the issue

        new_reals = reals
        new_attacks = []

        for file_name in sorted(cnt.keys()):

            num_files = cnt[file_name]

            try:
                sample_idx = random.sample(range(num_files), frames_perfile)
            except:
                sample_idx = range(num_files)

            ffs = [ff for ff in attacks if ff[0] == file_name]

            ffs = [ff for idx, ff in enumerate(ffs) if idx in sample_idx]
            new_attacks = new_attacks + ffs

        try:
            new_indexes = random.sample(range(len(attacks)), len(new_reals) - len(new_attacks))
        except:
            new_indexes = []

        new_attacks = new_attacks + [ff for idx, ff in enumerate(attacks) if idx in new_indexes]

        fnil_balanced = new_reals + new_attacks

    print("num_reals", len(new_reals), "num_attacks", len(new_attacks))

    return fnil_balanced


def get_index_from_file_names(file_names_and_labels, data_folder, max_samples_per_file):
    file_names_index_and_labels = []

    for file_name, label in file_names_and_labels:
        file_path = Path(data_folder) / file_name
        v = cv.VideoCapture(str(file_path))
        frames_in_video = int(v.get(cv.CAP_PROP_FRAME_COUNT))
        # print(f"{file_path} - {frames_in_video} frames")
        if max_samples_per_file > frames_in_video:
            indexes = [x for x in range(frames_in_video)]
        else:
            indexes = sample([x for x in range(frames_in_video)], max_samples_per_file)
        for k in indexes:
            file_names_index_and_labels.append((file_name, k, label))

    print(file_names_index_and_labels)
    return file_names_index_and_labels


def get_file_names_and_labels(files_folder, protocol_folder: Path = None, groups: List = None):
    """
    Get absolute names of the corresponding file objects and their class labels.

    **Parameters:**

    ``files`` : [File]
        A list of files objects defined in the High Level Database Interface
        of the particular datbase.

    ``data_folder`` : str
        A directory containing the training data.

    **Returns:**

    ``file_names_and_labels`` : [(str, int)]
        A list of tuples, where each tuple contain an absolute filename and
        a corresponding label of the class.
    """

    file_names_and_labels = []
    folder = Path(files_folder)
    files_present = []
    for g in groups:
        files = [x.name for x in (folder / g).iterdir() if x.suffix == ".avi"]
        files_present.extend(files)
    print(files_present)
    if protocol_folder is not None:
        for file in protocol_folder.iterdir():
            if any([x in file.name.lower() for x in groups]) and file.suffix == ".txt":
                group = [g for g in groups if g in file.name.lower()][0]
                with file.open("r") as f:
                    # print(file.name)
                    for line in f:
                        if line.isspace():
                            continue
                        label, name = line.split(",")
                        name = f"{name.strip()}.avi"
                        # print(f"{name.strip()} - {label}")
                        if name in files_present:
                            file_names_and_labels.append((f"{group}/{name}", int(label)))

    print(file_names_and_labels)
    return file_names_and_labels


# ==============================================================================
class DataFolderPixBiS(data.Dataset):

    def __init__(self, data_folder,
                 transform=None,
                 groups=['train', 'dev', 'eval'],
                 protocol='nowig',
                 purposes=['real', 'attack'],
                 allow_missing_files=True,
                 do_balance=False,
                 max_samples_per_file=10,
                 channels='RGB',
                 mask_op='flat',
                 custom_size=224,
                 protocol_folder=None,
                 **kwargs):
        """
        **Parameters:**

        ``data_folder`` : str
            A directory containing the training data.

        ``transform`` : callable
            A function/transform that  takes in a PIL image, and returns a
            transformed version. E.g, ``transforms.RandomCrop``. Default: None.

        ``extension`` : str
            Extension of the data files. Default: ".hdf5".
            Note: this is the only extension supported at the moment.

        ``bob_hldi_instance`` : object
            An instance of the HLDI interface. Only HLDI's of bob.pad.face
            are currently supported.

        ``hldi_type`` : str
            String defining the type of the HLDI. Default: "pad".
            Note: this is the only option currently supported.

        ``groups`` : str or [str]
            The groups for which the clients should be returned.
            Usually, groups are one or more elements of ['train', 'dev', 'eval'].
            Default: ['train', 'dev', 'eval'].

        ``protocol`` : str
            The protocol for which the clients should be retrieved.
            Default: 'grandtest'.

        ``purposes`` : str or [str]
            The purposes for which File objects should be retrieved.
            Usually it is either 'real' or 'attack'.
            Default: ['real', 'attack'].

        ``allow_missing_files`` : str or [str]
            The missing files in the ``data_folder`` will not break the
            execution if set to True.
            Default: True.
        """

        self.data_folder = Path(data_folder)
        self.modalities = 'color'
        self.transform = transform
        self.groups = groups
        self.protocol = protocol
        self.purposes = purposes
        self.allow_missing_files = allow_missing_files
        self.do_balance = do_balance
        self.max_samples_per_file = max_samples_per_file
        self.channels = channels
        self.mask_op = mask_op
        self.custom_size = custom_size

        protocol_folder = Path(protocol_folder) / protocol

        file_names_and_labels = get_file_names_and_labels(files_folder=data_folder,
                                                          protocol_folder=protocol_folder,
                                                          groups=groups)

        self.file_names_and_labels = file_names_and_labels  # list of videos # relative paths, not absolute

        # Added shuffling of the list

        file_names_index_and_labels = get_index_from_file_names(self.file_names_and_labels, self.data_folder,
                                                                self.max_samples_per_file)

        ## Subsampling should be performed here. Should respect the attack types?, here it subsamples based on file

        if self.do_balance:
            file_names_index_and_labels = balance_data_filebased(file_names_index_and_labels)

        random.shuffle(file_names_index_and_labels)

        self.file_names_index_and_labels = file_names_index_and_labels

    # ==========================================================================
    def __getitem__(self, index):

        # with index, figure out which file and which frame

        """
        Returns an image, possibly transformed, and a target class given index.

        **Parameters:**

        ``index`` : int.
            An index of the sample to return.

        **Returns:**

        ``pil_img`` : Tensor or PIL Image
            If ``self.transform`` is defined the output is the torch.Tensor,
            otherwise the output is an instance of the PIL.Image.Image class.

        ``target`` : int
            Index of the class.

        """
        path, frame_idx, target = self.file_names_index_and_labels[index]  # relative path

        modality_path = self.data_folder / path  # Now, absolute path to each of the files

        try:

            img_array = get_frame(modality_path, frame_idx=frame_idx)
            # print(img_array.shape)
            if img_array.shape[0] == 3:

                #  Even pixels in height and width

                if img_array.shape[1] % 2 != 0:
                    img_array = img_array[:, 0:-1, :]
                if img_array.shape[2] % 2 != 0:
                    img_array = img_array[:, :, 0:-1]

                img_array_rgb = img_array.copy()

                cv_rgb = np.moveaxis(img_array_rgb, 0, -1)

                img_array_tr = cv_rgb.copy()

            else:
                # np.moveaxis(img_array, 0, -1)
                img_array_tr = img_array.copy()

            pil_img = img_array_tr.copy()
            assert (img_array_tr.shape[1] == 224)
            stacker = pil_img.copy()
            stacker = np.array(stacker, dtype='uint8')  # uint8 is required otherwise the PIL.Image wont know what it is

            pil_imgs_stacko = stacker.copy()
            if self.transform is not None:
                pil_imgs_stacko = self.transform(pil_imgs_stacko)

            if target == -1:  # BF

                if self.mask_op == 'flat':

                    mask = np.ones((img_array_tr.shape[1], img_array_tr.shape[2]), dtype='float') * 0.99

                    if self.custom_size != 224:
                        mask = np.ones((self.custom_size, self.custom_size), dtype='float') * 0.99

            else:
                mask = np.ones((img_array_tr.shape[1], img_array_tr.shape[2]), dtype='float') * 0.01

                if self.custom_size != 224:
                    mask = np.ones((self.custom_size, self.custom_size), dtype='float') * 0.01

            labels = {'pixel_mask': Tensor(mask), 'binary_target': target}

            img = {'image': Tensor(pil_imgs_stacko)}
            # print(pil_imgs_stacko.shape)
            return img, labels
        except Exception as e:
            print(f"{path} - {frame_idx}")
            traceback.print_exc()

    # ==========================================================================
    def __len__(self):
        """
        **Returns:**

        ``len`` : int
            The length of the file list.
        """
        return len(self.file_names_index_and_labels)
