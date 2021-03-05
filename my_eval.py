import os
import glob
import argparse
import numpy as np
import tifffile
import cv2 as cv
import xlwt
import xlrd


def parse_args():
	parser = argparse.ArgumentParser(description='Compute depth errors.')
	parser.add_argument('--gt_path', type=str, default='/media/zyd/Elements/EndoVis/0original_all')
	parser.add_argument('--model_name', type=str, required=True)
	return parser.parse_args()


def compute_errors(gt, pred):
	gt = gt[:, :, 2]
	arr = np.resize(pred, (640, 512))
	arr = 1/arr
	arr = cv.resize(arr, (1280, 1024))
	mask = (gt > 1e-3)
	scale_factor = np.median(gt[mask])/np.median(arr[mask])
	arr = arr[mask]
	pred = arr*scale_factor
	gt = gt[mask]

	thresh = np.maximum((gt / pred), (pred / gt))
	a1 = (thresh < 1.25     ).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	rmse = (gt - pred) ** 2
	rmse = np.sqrt(rmse.mean())

	rmse_log = (np.log(gt) - np.log(pred)) ** 2
	rmse_log = np.sqrt(rmse_log.mean())

	abs_rel = np.mean(np.abs(gt - pred) / gt)

	sq_rel = np.mean(((gt - pred) ** 2) / gt)

	return rmse, rmse_log, abs_rel, sq_rel, a1, a2, a3, scale_factor


def find_tiff(gt_path, npy_path):
	_, filename = os.path.split(npy_path)
	id = filename[10:22]
	tiff_path = os.path.join(gt_path, "d{}".format(id[2]), "k{}".format(id[4]), "Depth/left_depth_map{}.tiff".format(id))
	return tiff_path

if __name__ == '__main__':
	args = parse_args()
	gt_path = args.gt_path
	pred_path = os.path.join("/media/zyd/Elements/OMEN Ubuntu backup/respository/monodepth2_results", args.model_name, "depth")
	
	pred_paths = glob.glob(os.path.join(pred_path, '*.{}'.format("npy")))
	print("-> Computing errors on {:d} test images".format(len(pred_paths)))

	errors = []
	for idx, npy_path in enumerate(pred_paths):
		pred = np.load(npy_path)
		tiff_path = find_tiff(gt_path, npy_path)
		gt = tifffile.imread(tiff_path)
		errors.append(compute_errors(gt, pred))
		print("   Computed errors {:d} of {:d} images".format(idx + 1, len(pred_paths)))
	print("-> Done! ^_^")

	errors = np.array(errors)
	from pandas.core.frame import DataFrame
	errors = DataFrame(errors)
	errors.dropna(axis=0, how='any')
	print(errors.mean(axis=0))
	# f = xlwt.Workbook()
	# sheet = f.add_sheet('sheet', cell_overwrite_ok = True)
	# excel_path = os.path.join(pred_path, "{}.xls".format(args.model_name))
	# f.save(excel_path)

	