import os
import numpy as np
import tifffile
import cv2 as cv
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm


def compute_error(gt, pred):
	gt = np.nan_to_num(gt[:, :, 2])
	gt = np.clip(gt, 0, np.percentile(gt[mask], 99.9))
	
	arr = np.resize(pred, (640, 512))
	arr = 1/arr
	arr = cv.resize(arr, (1280, 1024))

	mask = (gt > 1e-3)

	scale_factor = np.median(gt[mask])/np.median(arr[mask])
	pred = scale_factor*arr
	print("\nscale_factor:", scale_factor)
	error = np.abs(pred - gt)
	print(error)

	# import matplotlib.pyplot as plt
	# n, bins, patches = plt.hist(x = pred[mask])
	# plt.savefig('hist_d2k1_37pred.png')
	
	return error, mask

#写Excel
def draw_map(error, mask):
	vmax = np.percentile(error[mask], 99.9)
	vmin = np.percentile(error[mask], 0.1)
	print("error max: ", vmax)
	print("error min: ", vmin)
	norm_error = (error - vmin) / (vmax - vmin)
	norm_error[~mask] = 0
	mapper = cm.ScalarMappable(cmap='pink')
	colormapped_im = ((mapper.to_rgba(norm_error)[:, :, :3] * 255).astype(np.uint8))

	im = pil.fromarray(colormapped_im)
	name_dest_im = "60_errormap_left_d6k1.jpeg"
	im.save(name_dest_im)
    

if __name__ == '__main__':
	path = "/OMEN Ubuntu backup/respository/monodepth2_results"
	model = "60_0.05ZNCC16_OFFd6_mix_finetuneRes50"
	testset = "6"

	gt = []
	disp = []
	for i in range(1,6):
		gt.append(tifffile.imread(os.path.join(path, "keyframes/left_depth_map_d{}k{}.tiff".format(testset, i))))
		disp.append(np.load(os.path.join(path, model, "Left_Image_d{}k{}_disp.npy".format(testset, i))))
	
	gt.append(tifffile.imread(os.path.join(path, "keyframes/right_depth_map_d{}k5.tiff".format(testset))))
	disp.append(np.load(os.path.join(path, model, "Right_Image_d{}k5_disp.npy".format(testset))))

	id = 0
	error, mask = compute_error(gt[id], disp[id])
	draw_map(error, mask)