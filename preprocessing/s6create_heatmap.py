import numpy as np
import os
import cv2
from os.path import splitext,join,isfile
from utils import get_path_files,print_array,read_pointlabel
from densemap import Densemap
import win_unicode_console
win_unicode_console.enable()

#org width:3680->64/72
#org height:644->64/72
def potlabels_to_densemap(inlabelfile,outputdir,heatmap_size):
    #xscale=1./(400./800.)  #org width  400
    #yscale=1./(280./140.)  #org height 280
    xscale=3680.0/heatmap_size[1]
    yscale=644.0/heatmap_size[0]
    vs_pointslables=read_pointlabel(inlabelfile,xscale,yscale)
    count_num=0
    densemap=Densemap(image_size=heatmap_size,heatmap_size=heatmap_size)
    for pointslabel in vs_pointslables:
        filename=pointslabel["filename"]
        vspoints=pointslabel["vspoints"]
        if len(vspoints)==0:
            continue
        #if "aggTTmhdAqMXMii" not in filename:
        #    continue
        target=densemap.generate_target(np.array(vspoints))
        output=densemap.target_merge(target)
        filename_list=filename.split('/')
        filepre,fileext=splitext(filename_list[2])
        print('*',end=' ')
        np.savetxt(outputdir+filepre+'.csv', output, delimiter = ',')
        count_num+=1
    print("create image and matlabel:",count_num)

def nonlabels_to_densemap(inputdir,outputdir,heatmap_size):
    files=get_path_files(inputdir)
    print(len(files))
    count_num=0
    for eachfile in files:
        vspoints=[]
        filepre,fileext=splitext(eachfile)
        output=np.zeros((heatmap_size[0],heatmap_size[1]),dtype=np.float32) 
        print('*',end=' ')
        np.savetxt(outputdir+filepre+'.csv', output, delimiter = ',')
        count_num+=1
    print("create image densemap csv:",count_num)

if 1: #org data
    #heatmap_size=(72,72)
    heatmap_size=(56,88)
    outputdir="D:\\createset\\potholeDt_new\\test\\ground_truth_csv/"
    inputdir="E:\\@pothole_dataset\\pothole_orgroi\\test\\negative\\"
    nonlabels_to_densemap(inputdir,outputdir,heatmap_size)
    outputdir="D:\\createset\\potholeDt\\train\\ground_truth_new_csv/"
    inputdir="E:\\@pothole_dataset\\pothole_orgroi\\train\\negative\\"
    #nonlabels_to_densemap(inputdir,outputdir,heatmap_size)

if 1:
    #heatmap_size=(72,72)
    heatmap_size=(56,88)
    outputdir="D:\\createset\\potholeDt_new\\test\\ground_truth_csv_1/"
    #inlabelfile="E:\\@pothole_dataset\\pothole_org\\labels_positive_orgpots_test.txt"
    inlabelfile="E:\\@pothole_dataset\\Dataset(Simplex)\\labels_positive_broipots_test.txt"
    potlabels_to_densemap(inlabelfile,outputdir,heatmap_size)
    outputdir="D:\\createset\\potholeDt_new\\train\\ground_truth_csv_1/"
    #inlabelfile="E:\\@pothole_dataset\\pothole_org\\labels_positive_orgpots_train.txt"
    inlabelfile="E:\\@pothole_dataset\\Dataset(Simplex)\\labels_positive_broipots_train.txt"    
    potlabels_to_densemap(inlabelfile,outputdir,heatmap_size)