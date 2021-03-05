import os
import json
import numpy as np
import matplotlib.pyplot as plt

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs

gt_global_poses = []
for i in range(1, 81): 
    gt_poses_path = os.path.join("/media/zyd/Elements/EndoVis/0original_all/d3/k2", "frame_data", "frame_data{:06d}.json".format(i))
    with open(gt_poses_path, "r") as f:
        data = json.load(f)
        pose = data['camera-pose']
    gt_global_poses.append(pose)

gt_global_poses = np.array(gt_global_poses)
gt_xyzs = gt_global_poses[:, :3, 3]
gt_local_poses = []
for i in range(1, len(gt_global_poses)):
    gt_local_poses.append(
        np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))
print(np.array(gt_local_poses).shape)

pred_poses = np.load('/media/zyd/Elements//OMEN Ubuntu backup/respository/monodepth2_results/17_posebiLSTM_PSNR_3d_OFFd3/d3k2_poses.npy')
Lipred_poses = np.load('/media/zyd/Elements/OMEN Ubuntu backup/respository/monodepth2_results/25_poseLi_PSNR_3d_OFFd3/d3k2_poses.npy')
Opred_poses = np.load('/media/zyd/Elements/OMEN Ubuntu backup/respository/monodepth2_results0/78_5e-4PSNR_3d_OFFd3_mix_finetuneRes50/d3k2_poses.npy')

for i in range(79):
    local_xyzs = np.array(dump_xyz(pred_poses))
    Lilocal_xyzs = np.array(dump_xyz(Lipred_poses))
    Olocal_xyzs = np.array(dump_xyz(Opred_poses))
    gt_local_xyzs = np.array(dump_xyz(gt_local_poses))
print("local_xyzs:", local_xyzs.shape)
print("gt_local_xyzs:", gt_local_xyzs.shape)
scale = np.sum(gt_local_xyzs * local_xyzs) / np.sum(local_xyzs ** 2)
Oscale = np.sum(gt_local_xyzs * Olocal_xyzs) / np.sum(Olocal_xyzs ** 2)
Liscale = np.sum(gt_local_xyzs * Lilocal_xyzs) / np.sum(Lilocal_xyzs ** 2)
local_xyzs = scale*local_xyzs
Olocal_xyzs = Oscale*Olocal_xyzs
Lilocal_xyzs = Liscale*Lilocal_xyzs

ax = plt.axes(projection='3d')
plt.title("Visual odometry", fontsize=15)
ax.view_init(35, 40)
# ax.scatter3D(xdata, ydata, zdata, c = zdata, cmap = 'viridis')
ax.scatter3D(gt_local_xyzs[:, 0], gt_local_xyzs[:, 1], gt_local_xyzs[:, 2], color = 'r', label='Groundtruth')
ax.scatter3D(local_xyzs[:, 0], local_xyzs[:, 1], local_xyzs[:, 2], color = 'b', label='BiLSTM')
ax.scatter3D(Lilocal_xyzs[:, 0], Lilocal_xyzs[:, 1], Lilocal_xyzs[:, 2], color = 'black', label='Li Ning')
ax.scatter3D(Olocal_xyzs[:, 0], Olocal_xyzs[:, 1], Olocal_xyzs[:, 2], color = 'g', label='Backbone')
plt.legend()
plt.show()