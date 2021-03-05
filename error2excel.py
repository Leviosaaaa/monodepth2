import os
import numpy as np
import tifffile
import cv2 as cv
import xlwt
import xlrd

def compute_error(gt, pred):

	gt = np.nan_to_num(gt[:, :, 2])
	gt = np.clip(gt, 0, np.percentile(gt, 99.9))
	
	arr = np.resize(pred, (640, 512))
	arr = 1/arr
	print("\nscaled depth: ", arr)
	arr = cv.resize(arr, (1280, 1024))

	mask = (gt > 1e-3)

	scale_factor = np.median(gt[mask])/np.median(arr[mask])
	pred = scale_factor*arr[mask]
	gt = gt[mask]
	print("\nscale_factor:", scale_factor)
	# print("gt's mean:", np.mean(gt))
	# print("pred's mean:", np.mean(pred))



	RMSE, logR, AbsRel, SqRel = 0, 0, 0, 0
	a1, a2, a3 = 0, 0, 0
	 
	thresh = np.maximum((gt / pred), (pred / gt))
	a1 = (thresh < 1.25     ).mean()
	a2 = (thresh < 1.25 ** 2).mean()
	a3 = (thresh < 1.25 ** 3).mean()

	RMSE = (gt - pred) ** 2
	RMSE = np.sqrt(RMSE.mean())

	logR = (np.log(gt) - np.log(pred)) ** 2
	logR = np.sqrt(logR.mean())

	AbsRel = np.mean(np.abs(gt - pred) / gt)

	SqRel = np.mean(((gt - pred) ** 2) / gt)

	print("RMSE = ", RMSE)
	print("logR = ", logR)
	print("AbsRel = ", AbsRel)
	print("SqRel = ", SqRel)
	print("1.25 percentage: ", a1)
	print("1.25^2 percentage: ", a2)
	print("1.25^3 percentage: ", a3)
	return RMSE, logR, AbsRel, SqRel, a1, a2, a3, scale_factor

#写Excel
def write_excel(row, errors1, errors2, errors3, errors4, errors5, errors6):
    f = xlwt.Workbook()
    # f = xlrd.open_workbook(filename='save.xls')
    # sheet = f.get_sheet(0)
    sheet = f.add_sheet('sheet',cell_overwrite_ok=True)
    # colum0 = []
    for i in range(0,len(errors1)):      #写行
        sheet.write(row, i+1, str(errors1[i]))
    for i in range(0,len(errors2)):      #写行
        sheet.write(row+1, i+1, str(errors2[i]))
    for i in range(0,len(errors3)):      #写行
        sheet.write(row+2, i+1, str(errors3[i]))
    for i in range(0,len(errors4)):      #写行
        sheet.write(row+3, i+1, str(errors4[i]))
    for i in range(0,len(errors5)):      #写行
        sheet.write(row+4, i+1, str(errors5[i]))
    for i in range(0,len(errors6)):      #写行
        sheet.write(row+5, i+1, str(errors6[i]))
    #写第一列
    # for i in range(0,len(colum0)):
    #     sheet1.write(i+1,0,colum0[i])
    f.save('/media/zyd/Elements/OMEN Ubuntu backup/respository/monodepth2_results/15_pose1000LSTM_PSNR_3d_OFFd2/15_pose1000LSTM_PSNR_3d_OFFd2.xls')
    print("Successfully wrote a set of errors.")

if __name__ == '__main__':
	path = "/media/zyd/Elements/OMEN Ubuntu backup/respository/monodepth2_results"
	model = "15_pose1000LSTM_PSNR_3d_OFFd2"
	testset = "2"

	gt = []
	disp = []
	for i in range(1, 6): 
		gt.append(tifffile.imread(os.path.join(path, "keyframes/left_depth_map_d{}k{}.tiff".format(testset, i))))
		disp.append(np.load(os.path.join(path, model, "Left_Image_d{}k{}_disp.npy".format(testset, i))))
	
	gt.append(tifffile.imread(os.path.join(path, "keyframes/right_depth_map_d{}k5.tiff".format(testset))))
	disp.append(np.load(os.path.join(path, model, "Right_Image_d{}k5_disp.npy".format(testset))))

	# print("disp[3]: ", disp[3][0, 0, :, :].shape)
	# print("gt[3][:, :, 2]: ", gt[3][:, :, 2].shape)
	# np.savetxt("d3k4_gt.txt", gt[3][:, :, 2])
	# np.savetxt("d3k4_disp.txt", disp[3][0, 0, :, :])

	write_excel(1, compute_error(gt[0], disp[0]), compute_error(gt[1], disp[1]), compute_error(gt[2], disp[2]), \
		           compute_error(gt[3], disp[3]), compute_error(gt[4], disp[4]), compute_error(gt[5], disp[5]))