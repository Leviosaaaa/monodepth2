from __future__ import absolute_import, division, print_function

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from layers import transformation_from_parameters
from utils import readlines
from options import MonodepthOptions
from datasets import endoOdomDataset
import networks


def parse_args():
    parser = argparse.ArgumentParser(description='Compute pose errors.')
    parser.add_argument('--gt_path', type=str, default='/media/zyd/Elements/EndoVis/0original_all')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--testset', type=str, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--data_path", type=str, default=os.path.join(os.path.dirname(__file__), "endo"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--num_layers", type=int,
                                 help="number of resnet layers",
                                 default=50,  # original: 18
                                 choices=[18, 34, 50, 101, 152])
    return parser.parse_args()


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    # Evaluate odometry on EndoVis
    filenames = readlines(os.path.join(
                          os.path.dirname(__file__), "splits", "odom", "{}.txt".format(opt.testset)))

    dataset = endoOdomDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False)

    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    model_path = os.path.join("/media/zyd/Elements/OMEN Ubuntu backup/monodepth2_models", opt.model_name, "models/weights_19")
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")

    pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in opt.frame_ids], 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation = pose_decoder(features)

            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)
    print("pred_poses: ", pred_poses.shape)

    gt_global_poses = []
    for i in range(1, 80): 
        gt_poses_path = os.path.join(opt.gt_path, "d{}".format(opt.testset[1]),
                                     "k{}/frame_data".format(opt.testset[3]), 
                                     "frame_data{:06d}.json".format(i))
        with open(gt_poses_path, "r") as f:
            data = json.load(f)
            pose = data['camera-pose']
        gt_global_poses.append(pose)

    gt_global_poses = np.array(gt_global_poses)
    # print("gt_global_poses: ", gt_global_poses)
    print("gt_global_poses: ", gt_global_poses.shape)
    # gt_global_poses = np.concatenate(
    #     (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    # gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 2
    for i in range(0, num_frames - 4):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.4f}, std: {:0.4f}\n".format(np.mean(ates), np.std(ates)))

    pose_name = opt.testset + "_poses.npy"
    save_path = os.path.join("/media/zyd/Elements/OMEN Ubuntu backup/respository/monodepth2_results", opt.model_name, pose_name)
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(parse_args())
