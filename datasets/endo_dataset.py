"""
Author: ZYD
Date: 2020/12/02
"""

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
import tifffile


class endoDataset(MonoDataset):
    """Superclass for endoscopy dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(endoDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.809, 0, 0.466, 0],
                           [0, 1.011, 0.508, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (640, 512)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        return True

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:07d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        # print("frame_index", frame_index)
        # depth_filename = os.path.join(self.data_path, folder, "depthmaps/{:07d}.npy".format(int(frame_index)))
        depth_gt = tifffile.imread('/home/zyd/respository/sfmlearner_results/cache/left_depth_map_d3k1_000000.tiff')
        # depth_gt = np.load(depth_filename)
        depth_gt = depth_gt[:, :, 2]
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt