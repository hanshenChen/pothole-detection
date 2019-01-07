import os
from os.path import splitext
from s3org_to_roi import org_to_roiimg,org_to_roilabel
import win_unicode_console
win_unicode_console.enable()

if 1:
    #num=org_to_roiimg("E:\\@pothole_dataset\\Dataset(Simplex)\\test\\negative\\","E:\\@pothole_dataset\\Dataset(Simplex)_roi\\test\\negative\\")
    #print("finish roiimg invert num:",num)
    #num=org_to_roiimg("E:\\@pothole_dataset\\Dataset(Simplex)\\test\\positive\\","E:\\@pothole_dataset\\Dataset(Simplex)_roi\\test\\positive\\")
    #print("finish roiimg invert num:",num)
    #num=org_to_roiimg("E:\\@pothole_dataset\\Dataset(Simplex)\\train\\negative\\","E:\\@pothole_dataset\\Dataset(Simplex)_roi\\train\\negative\\")
    #print("finish roiimg invert num:",num)
    num=org_to_roiimg("E:\\@pothole_dataset\\Dataset(Simplex)\\train\\negative\\","E:\\@pothole_dataset\\Dataset(Simplex)_roi\\train\\negative\\")
    print("finish roiimg invert num:",num)
if 0:
    inpath="E:\\@pothole_dataset\\Dataset(Simplex)\\"
    in_txtfile='E:\\@pothole_dataset\\Dataset(Simplex)\\labels_positive_categorical_test_rp.txt'
    out_txtfile='E:\\@pothole_dataset\\Dataset(Simplex)\\btest_labels_positive_roiboxs.txt'
    #in_txtfile='E:\\@pothole_dataset\\pothole_org\\labels_positive_categorical.txt'
    #out_txtfile='E:\\@pothole_dataset\\pothole_org\\btrain_labels_positive_roiboxs.txt'
    org_to_roilabel(in_txtfile,out_txtfile,inpath,1.0,1.0)