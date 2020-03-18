import os
import numpy as np
import sys
from models.submodel import create_lcnn_model
from models.patchnet import PatchNet
from models.location_aware_model import Location_aware_model
from utils.data_loader import ImageDataLoader

def eval_clss(gt_clss, pr_clss):
    SMOOTH = 1e-12
    threshold = 0.5
    TP, FN, FP, TN = .0, .0, .0, .0
    for i, (gt, pr) in enumerate(zip(gt_clss, pr_clss)):
        if gt != 0:
            TP += pr >= threshold
            FN += pr < threshold
        else:
            FP += pr >= threshold
            TN += pr < threshold

    recall = TP / (TP + FN + SMOOTH)
    precision = TP / (TP + FP + SMOOTH)
    f11 = 2 * TP / (2 * TP + FP + FN + SMOOTH)
    print(f11)
    f1 = 2.0 * (precision * recall) / (precision + recall + SMOOTH)
    accuracy = (TP + TN) / (TP + FP + FN + TN + SMOOTH)
    print("Acc=%.4f,Recall=%.4f,Precision=%.4f,F1=%.4f" % (accuracy, recall, precision, f1))

home_path=os.path.expanduser("~")
resize_data_bpath=os.path.join(home_path,'dataset/pothole_resizeroi')
test_img_path = os.path.join(resize_data_bpath,'test/images-all')
test_gtdes_path = os.path.join(resize_data_bpath,'test/ground_truth_csv')#if not have ground truth ,gtdes=None

data_loader_test = ImageDataLoader(resize_data_bpath,img_path=test_img_path, gtdes_path=test_gtdes_path, btrain=False)
img_size = data_loader_test.get_img_size()

patch_data_path =os.path.join(home_path,"dataset/patch_new224")
orgroi_data_path=os.path.join(home_path,"dataset/Dataset(Simplex)_roi/test/images")

# create and load classification model
weightsfile='./model_weights/rn50_patch_cls3.h5'
patchnet=PatchNet(patch_data_path,weightsfile)

# create and load localization model
weightsfile='./model_weights/lcnn_pothole_100.h5'
lcnn_model=create_lcnn_model(img_size+(3,))
lcnn_model.load_weights(weightsfile,by_name=True)

evaluate_model= Location_aware_model(lcnn_model,patchnet,orgroi_data_path)
evaluate_model.set_superparm(2,2)

gt_dess = data_loader_test.blob_gtdes
gt_clss = data_loader_test.blob_gtcls
org_files = data_loader_test.data_files
in_imgs = data_loader_test.imagenet_mean(data_loader_test.blob_img)
pr_dess,pr_clss = evaluate_model.predict_all(in_imgs,org_files)
des_mae = np.mean(np.square(pr_dess - gt_dess))
des_mse = np.mean(np.abs(pr_dess - gt_dess))
print(des_mse, des_mae)

eval_clss(gt_clss, pr_clss)

