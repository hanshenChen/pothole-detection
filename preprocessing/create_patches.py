import numpy as np
import cv2
import os
import random as rand
import time
import math
from os.path import splitext,join,isfile
from utils import get_path_files,get_boundingbox,read_boxlabel,rect_topoint
import win_unicode_console
win_unicode_console.enable()
patchsize=224

def find_negative_rects(posrect,boxs,patchSize,img_width,img_height):
    adjacency_matrix=[]
    adjacency_matrix.append((posrect[0]-patchSize,posrect[1]-patchSize))
    adjacency_matrix.append((posrect[2],posrect[1]-patchSize))
    adjacency_matrix.append((posrect[0]-patchSize,posrect[3]))
    adjacency_matrix.append((posrect[2],posrect[3]))
    rand.seed(time.time())
    #48*48    offsetx=0-192, offsety=0-96
    #224*224  offsetx=0-896, offsety=0-448
    rand_add=rand.randint(0,patchSize)
    adjacency_matrix.append((img_width/2-rand_add,img_width-patchSize-rand_add/2))
    adjacency_matrix.append((img_width/2+rand_add,img_width-patchSize-rand_add/2))
    rand_add=rand.randint(0,patchSize)
    adjacency_matrix.append((img_width/2-rand_add,img_width-patchSize-rand_add/2))
    adjacency_matrix.append((img_width/2+rand_add,img_width-patchSize-rand_add/2))
    rand_add=rand.randint(patchSize,patchSize*2)
    adjacency_matrix.append((img_width/2-rand_add,img_width-patchSize-rand_add/2))
    adjacency_matrix.append((img_width/2+rand_add,img_width-patchSize-rand_add/2))
    rand_add=rand.randint(patchSize,patchSize*2)
    adjacency_matrix.append((img_width/2-rand_add,img_width-patchSize-rand_add/2))
    adjacency_matrix.append((img_width/2+rand_add,img_width-patchSize-rand_add/2))
    rand_add=rand.randint(patchSize*2,patchSize*3)
    adjacency_matrix.append((img_width/2-rand_add,img_width-patchSize-rand_add/2))
    adjacency_matrix.append((img_width/2+rand_add,img_width-patchSize-rand_add/2))
    rand_add=rand.randint(patchSize*2,patchSize*3)
    adjacency_matrix.append((img_width/2-rand_add,img_width-patchSize-rand_add/2))
    adjacency_matrix.append((img_width/2+rand_add,img_width-patchSize-rand_add/2))
    rand_add=rand.randint(patchSize*3,patchSize*4)
    adjacency_matrix.append((img_width/2-rand_add,img_width-patchSize-rand_add/2))
    adjacency_matrix.append((img_width/2+rand_add,img_width-patchSize-rand_add/2))
    rand_add=rand.randint(patchSize*3,patchSize*4)
    adjacency_matrix.append((img_width/2-rand_add,img_width-patchSize-rand_add/2))
    adjacency_matrix.append((img_width/2+rand_add,img_width-patchSize-rand_add/2))
    for mi in adjacency_matrix:
        x1=mi[0]
        y1=mi[1]
        x2=mi[0]+patchSize
        y2=mi[1]+patchSize
        if x1<0 or x1>=img_width or x2<0 or x2>=img_width or y1<0 or y1>=img_height or y2<0 or y2>=img_height:
            continue
        ai=(x1,y1,x2,y2)
        count=0
        for box in boxs:
            if intersection(ai,box)==0:
                count+=1
        if len(boxs)==count:
            return int(x1),int(y1),int(x2),int(y2)
    print("can't find box area")
    return(0,0,0,0)

def intersection(ai,bi):
    # ai and bi should be x1,y1,x2,y2
    x=max(ai[0],bi[0])
    y=max(ai[1],bi[1])
    w=min(ai[2],bi[2])-x
    h=min(ai[3],bi[3])-y
    if w<0 or h<0:
        return 0
    return w*h

def get_rand_offset_point(p,patchsize):
    ps=[]
    ps.append((p[0],p[1]))
    offset_pixel=math.ceil(patchsize/4)
    rand.seed(time.time())
    for i in [[0,1.0/2],[1.0/4,1]]:
        offsetx=rand.randint(offset_pixel*i[0],offset_pixel*i[1])
        offsety=rand.randint(offset_pixel*i[0],offset_pixel*i[1])
        print(offsetx,offsety)
        #ps.append((p[0],p[1]+offsety))
        #ps.append((p[0],p[1]-offsety))
        #ps.append((p[0]+offsetx,p[1]))
        ps.append((p[0]+offsetx,p[1]+offsety))
        ps.append((p[0]+offsetx,p[1]-offsety))

        #ps.append((p[0]-offsetx,p[1]))
        ps.append((p[0]-offsetx,p[1]+offsety))
        ps.append((p[0]-offsetx,p[1]-offsety))

    return ps

def find_positive_rects(ps,boxsize,img_width,img_height):
    rects=[]
    for p in ps:
        x1,y1,x2,y2=get_boundingbox(p,patchsize,img_width,img_height)
        rects.append((x1,y1,x2,y2))
    print(len(rects),end="")
    rects=list(set(rects))
    print(len(rects))
    return rects

def negset_create_patch(inputdir,outputpath,typefile,img_resize=None):
    subdir="negative/"
    count_num=0
    files=get_path_files(inputdir+typefile+"/"+subdir)
    for j,eachfile in enumerate(files):
        filepre,fileext=splitext(eachfile)
        imgfile=inputdir+typefile+"/"+subdir+eachfile
        #if "AApPkxlfOdiMjMs" not in imgfile:
        #    continue
        img=cv2.imread(imgfile)
        if img.any()==None:
            print("read image %s file failed",imgfile)
        if img_resize!=None:
            img=cv2.resize(img,img_resize)
        for i in range(20+10):
            rand.seed(time.time())
            rand_x=rand.randint(0,img.shape[1]-patchsize)  
            rand_y=rand.randint(0,img.shape[0]-patchsize) 
            p=(rand_x,rand_y) 
            print(p)
            x1,y1,x2,y2=get_boundingbox(p,patchsize,img.shape[1],img.shape[0])
            #cv2.rectangle(img, (x1,y1),(x2,y2),(0, 255, 0),1)
            roiImg = img[y1:y2,x1:x2] #利用numpy中的数组切片设置ROI区域
            filename= outputpath+typefile+"/"+subdir+filepre+str(i)+fileext
            cv2.imwrite(filename,roiImg)
            print('finshed write:',filename)
            count_num+=1
    print("created negative pathch:",count_num)

def posset_create_patch(inlabelfile,inputpath,outputpath,typefile,xscale,yscale,img_resize=None):
    pcount_num=0
    ncount_num=0
    boxslabels=read_boxlabel(inlabelfile,xscale,yscale)
    for boxslabel in boxslabels:
        filename=boxslabel["filename"]
        rects=boxslabel["vsrects"]
        imgfile=inputpath+filename
        if len(rects)==0:
            continue
        #if "AApPkxlfOdiMjMs.JPG" not in imgfile:
        #    continue
        img=cv2.imread(imgfile)
        xcord_rate=1.0
        ycord_rate=1.0
        if img is None:
            print("read image %s file failed",imgfile)
            continue
        if img_resize!=None:
            xcord_rate=img_resize[0]/img.shape[1]
            ycord_rate=img_resize[1]/img.shape[0]
            img=cv2.resize(img,img_resize)
        filename_list=filename.split('/')
        filepre,fileext=splitext(filename_list[2])
        boxs=[]
        cout=0
        for rect in rects:   
            p=rect_topoint(rect)
            p=list(p)
            p[0]=round(p[0]*xcord_rate)
            p[1]=round(p[1]*ycord_rate)
            x1,y1,x2,y2=get_boundingbox(p,patchsize,img.shape[1],img.shape[0])
            #cv2.rectangle(img, (x1,y1),(x2,y2),(0, 255, 0),1)
            #roiImg = img[y1:y2,x1:x2] #利用numpy中的数组切片设置ROI区域  
            #cv2.imwrite(outputpath+typefile+"/positive/"+filepre+str(i)+fileext,roiImg)
            boxs.append((x1,y1,x2,y2))

            ps=get_rand_offset_point(p,patchsize)
            rects=find_positive_rects(ps,patchsize,img.shape[1],img.shape[0])
            for rect in rects:
                x1,y1,x2,y2=rect
                #cv2.rectangle(img, (x1,y1),(x2,y2),(0, 255, 0),1)
                roiImg = img[y1:y2,x1:x2] #利用numpy中的数组切片设置ROI区域  
                cout+=1
                cv2.imwrite(outputpath+typefile+"/positive/"+filepre+str(cout)+fileext,roiImg)

        pcount_num +=cout
        for i,box in enumerate(boxs):
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            rect=(x1,y1,x2,y2)
            x1,y1,x2,y2=find_negative_rects(rect,boxs,patchsize,img.shape[1],img.shape[0])
            if x1==0 and y1==0 and x2==0 and y2==0:
                continue
            #cv2.rectangle(img, (x1,y1), (x2,y2),(0, 0, 255),1)
            roiImg = img[y1:y2,x1:x2] #利用numpy中的数组切片设置ROI区域  
            cv2.imwrite(outputpath+typefile+"/negative/"+filepre+str(i)+fileext,roiImg)
            ncount_num+=1

        print("created positive patch:%s, negative patch:%s" %(pcount_num,ncount_num))

typefile="train"
if 0:#org
    inputpath="E:\\@pothole_dataset\\pothole_orgroi-rp\\"
    outputpath="D:/createset/patch_t48/"
    negset_create_patch(inputpath,outputpath,typefile)
    labelfile="E:\\@pothole_dataset\\pothole_org\\"+typefile+'_labels_positive_roiboxs.txt'
    xscale=1.0
    yscale=1.0
    posset_create_patch(labelfile,inputpath,outputpath,typefile,xscale,yscale)

if 1:#big
    inputpath="E:\\@pothole_dataset\\Dataset(Simplex)_roi\\"#"E:\\PotholeDetection-master\\data\\"
    outputpath="D:/createset/patch_new224/"
    img_resize=(1760,1120)#(1840,1288)#(400,280)
    negset_create_patch(inputpath,outputpath,typefile,img_resize)
    labelfile="E:\\@pothole_dataset\\Dataset(Simplex)\\labels_positive_broiboxs_"+typefile+'.txt'
    xscale=1.0
    yscale=1.0
    #posset_create_patch(labelfile,inputpath,outputpath,typefile,xscale,yscale,img_resize)
cv2.waitKey()