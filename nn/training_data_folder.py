#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: ageorge
"""

# ==============================================================================
# Import what is needed here:

from collections import Counter
import random
import traceback
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np
import torch.utils.data as data
from torch import Tensor

from nn.common import get_coordinates_of_eyes_in_frame, get_slices

random.seed(a=7)


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


def balance_sample(bigger: List[Tuple[str, int]], smaller: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    names = [ff[0].split("__")[0] for ff in bigger]

    cnt = Counter(names)

    frames_per_file = round(len(bigger) / len(cnt))  # would be low, now add some more to solve the issue

    bigger_reduced = []

    for file_name in sorted(cnt.keys()):

        num_frames = cnt[file_name]

        try:
            sample_indexes = random.sample(range(num_frames), frames_per_file)
        except ValueError:
            sample_indexes = range(num_frames)

        file_tuples = [x for x in bigger_reduced if x[0].split("__")[0] == file_name]

        ffs = [x for i, x in enumerate(file_tuples) if i in sample_indexes]
        bigger_reduced.extend(ffs)

    try:
        new_indexes = random.sample(range(len(bigger)), len(smaller) - len(bigger_reduced))
    except ValueError:
        new_indexes = []

    bigger_reduced.extend([x for i, x in enumerate(bigger) if i in new_indexes])

    return bigger_reduced + smaller


def balance_data_file_based(tuples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:

    ''' Return a balanced list of files tuples '''
    reals = [ff for ff in tuples if ff[1] == 1]
    attacks = [ff for ff in tuples if ff[1] == -1]

    num_reals = len(reals)
    num_attacks = len(attacks)

    print("num_reals", len(reals), "num_attacks", len(attacks))

    # Simplest, always downsample, no upsampling here, from each file select a number of frames so as to balance with
    # the other class.

    if num_reals > num_attacks:

        # downsample reals
        balanced = balance_sample(reals, attacks)

    else:
        # downsample attacks        
        balanced = balance_sample(attacks, reals)

    new_reals = [ff for ff in balanced if ff[1] == 1]
    new_attacks = [ff for ff in balanced if ff[1] == -1]
    print("num_reals", len(new_reals), "num_attacks", len(new_attacks))

    return balanced


def get_file_names_and_labels(files_folder, partition, protocol_folder: Path, groups: List[str] = None) -> List[
    Tuple[str, int]]:
    """
    Get absolute names of the corresponding file objects and their class labels.

    **Returns:**

    ``file_names_and_labels`` : [(str, int)]
        A list of tuples, where each tuple contain an absolute filename and
        a corresponding label of the class.
    """

    file_names_and_labels = []
    folder = Path(files_folder)
    for g in groups:
        group_files = [x.name for x in (folder / g).iterdir() if x.suffix == ".png"]
        partition_data = protocol_folder / f"{g.capitalize()}_{partition}.txt"
        with partition_data.open("r") as f:
            for line in f:
                if line.isspace():
                    continue
                label, name = line.split(",")
                name = name.strip()
                item_files = [x for x in group_files if x.startswith(name)]
                item_tuples = [(f"{g}/{x}", int(label)) for x in item_files]
                file_names_and_labels.extend(item_tuples)
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
                 partition=1,
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
                                                          partition=partition,
                                                          protocol_folder=protocol_folder,
                                                          groups=groups)

        if self.do_balance:
            file_names_and_labels = balance_data_file_based(file_names_and_labels)

        # Added shuffling of the list
        random.shuffle(file_names_and_labels)

        self.file_names_and_labels = file_names_and_labels  # list of videos # relative paths, not absolute

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
        path, target = self.file_names_and_labels[index]  # relative path
        complete_path = self.data_folder / path  # Now, absolute path to each of the files
        try:

            img_array = cv.imread(str(complete_path))
            print(img_array)
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
            print(f"{path}")
            traceback.print_exc()

    # ==========================================================================
    def __len__(self):
        """
        **Returns:**

        ``len`` : int
            The length of the file list.
        """
        return len(self.file_names_index_and_labels)
