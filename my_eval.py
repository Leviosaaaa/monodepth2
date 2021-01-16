import os
import glob
import numpy as np
import tifffile
import cv2 as cv
import xlwt
import xlrd


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

#写Excel
def write_excel(row, errors, pred_name, gt_name):
    f = xlwt.Workbook()
    sheet = f.add_sheet('sheet',cell_overwrite_ok=True)
    for i in range(0,len(errors)):      #写行
        sheet.write(row, i+1, str(errors[i]))
    sheet.write(row, len(errors)+1, pred_name)
    sheet.write(row, len(errors)+2, gt_name)
    f.save('test.xls')
    print("Successfully wrote a set of errors.")

if __name__ == '__main__':
	gt_path = "/home/zyd/respository/monodepth2_results/keyframes"
	pred_path = "/home/zyd/respository/monodepth2_results/20_OFFd2_mix_finetuneRes50_nomask"
	out_path = "/home/zyd/respository/monodepth2_results/20_OFFd2_mix_finetuneRes50_nomask"
	
	pred_paths = glob.glob(os.path.join(pred_path, '*.{}'.format("npy")))
	
	# if (len(pred_paths) != len(gt_paths)):
	# 	print("-> SOS")
	# else:
	# 	print("-> Predicting on {:d} test images".format(len(pred_paths)))
	print("-> Predicting on {:d} test images".format(len(pred_paths)))
	
	f = xlwt.Workbook()
	sheet = f.add_sheet('sheet',cell_overwrite_ok=True)

	for idx, npy_path in enumerate(pred_paths):
		pred = np.load(npy_path)
		
		(filepath, filename) = os.path.split(npy_path)
		tiff_path = os.path.join(gt_path, "left_depth_map_d3k1_{}.tiff".format(filename[16:22]))
		(tiffpath, tiffname) = os.path.split(tiff_path)
		gt = tifffile.imread(tiff_path)

		errors = compute_errors(gt, pred)
		for i in range(0,len(errors)):      #写行
			sheet.write(idx, i+1, str(errors[i]))
		sheet.write(idx, len(errors)+1, filename)
		sheet.write(idx, len(errors)+2, tiffname)

		print("   Computed errors {:d} of {:d} images".format(idx + 1, len(pred_paths)))
	
	f.save('test.xls')
	print("-> Done!")
	