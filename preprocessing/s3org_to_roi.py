import numpy as np
import cv2
import os
from utils import printdim,print_array,get_path_files,create_norml_box_txt,read_boxlabel
import win_unicode_console
win_unicode_console.enable()

def sum_horizontal_pixel(inImg, bDebug):
    outImg=np.zeros(inImg.shape[0], dtype='int32')
    testImg=np.zeros((inImg.shape[0],255), dtype='int8')
    for i in range(inImg.shape[0]):
        for j in range(inImg.shape[1]):
            outImg[i] += inImg[i,j]
        cv2.line(testImg, (255,i), (255 - int(outImg[i]/inImg.shape[1]),i), 255, 1, 8);
    if (bDebug == True):
        cv2.imshow("testImg histogram", testImg)
    return outImg

#return the cut y Coordinate 
def get_roi_startY(inImg):
    grayimg=cv2.cvtColor(inImg,cv2.COLOR_BGR2GRAY)
    k=1.0
    if(grayimg.shape[0]>600 and grayimg.shape[1]>800):
        grayimg=cv2.resize(grayimg,(800,600))
        k=4.6
    #printdim(inputImg)
    #cv2.imshow("input",inputImg)
    rows=grayimg.shape[0]
    cols=grayimg.shape[1]
    roiImg = grayimg[int(rows*0.5):int(rows*(0.5+0.3)),int(cols*0.3):int(cols*(0.3+0.4))]
    #printdim(roiImg)
    #cv2.imshow("roi",roiImg)
    hstImg=sum_horizontal_pixel(roiImg, False)
    a=np.min(hstImg)   #最暗的地方
    y= np.where(hstImg==a)
    cuty= y[0][0]
    cuty=cuty-20 + int(grayimg.shape[0]*0.5);
    cuty1=int((cuty-140)*k)
    cuty2=int(cuty*k)
    return cuty1,cuty2

def disp_imgroi(inpath):
    files=get_path_files(inpath)
    print(len(files))
    for file in files:
        image = cv2.imread(inpath+file,0)
        if (image.all()==None):
            print("failed to read file")
        y=get_roi_startY(image)
        print(y,file)
        imgbk=image.copy()
        cv2.rectangle(imgbk, (0, y - 140), (image.shape[1], y), (0,255, 0), 2, 8, 0);
        cv2.imshow("image", imgbk);
        cv2.waitKey()

def cut_roi(inimg):
    y1,y2=get_roi_startY(inimg)
    roi_img=inimg[y1:y2,:,:]
    return roi_img

def org_to_roiimg(inpath,outpath):
    files=get_path_files(inpath)
    print(len(files))
    count_num=0
    for i,file in enumerate(files):
        image = cv2.imread(inpath+file)
        if (image.all()==None):
            print("failed to read %s file",file)
            continue
        roi_img=cut_roi(image)
        count_num+=1
        print(i,"create:",outpath+file)
        cv2.imwrite(outpath+file,roi_img)
    return count_num

#read,roi,write
def org_to_roilabel(in_txtfile,out_txtfile,in_imgpath,xscale=4.6,yscale=4.6):
    vs_boxlabels=[]
    count_num=0
    fileslabels=read_boxlabel(in_txtfile,xscale,yscale)
    for filelabels in fileslabels:
        filename=filelabels["filename"]
        rects=filelabels["vsrects"]
        imgfile=in_imgpath+filename
        if(os.path.exists(imgfile)==False):
            print("cann't find %s",imgfile)
        img=cv2.imread(imgfile)
        if img.any()==None:
            print("read image %s file failed",imgfile)
        count_num+=1
        vs_rects=[]
        for rect in rects:
            x=rect[0]
            y1,y2=get_roi_startY(img)
            y=rect[1]-y1
            width=rect[2]
            height=rect[3]
            rect=(x,y,width,height)
            vs_rects.append(rect)
        image_info={"filename":filename,"vsrects":vs_rects}
        print(image_info["filename"],image_info["vsrects"])
        vs_boxlabels.append(image_info)
    create_norml_box_txt(out_txtfile,vs_boxlabels)
    return  count_num

#for test
#inPath = "E:\\@pothole_dataset\\pothole_org\\data\\train\\positive\\";
#disp_imgroi(inPath)
