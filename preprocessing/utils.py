import os

#============================base====================
def print_array(arr,tag="*"):
    print(tag)
    for rows in arr:
        print("[",end="")
        for col in rows:
            #for a in col:
                print("%.4f" %(col),end=" ")
        print("]",end="")
        print("")

def printdim(inimg):
    for i in range(inimg.ndim):
        print(inimg.shape[i],end=",")
    print("")

def get_path_files(filepath):
    files=[]
    pathDir=os.listdir(filepath)
    pathDir.sort(key=lambda x: x[:-4].lower(),reverse=True)
    for eachfile in pathDir:
        #file=os.path.join('%s\%s' %(filepath,eachfile))
        #print(eachfile)
        files.append(eachfile)
    return files

def rect_topoint(rect):
    px = round(rect[0]+(rect[2])/2)
    py = round(rect[1]+(rect[3])/2)
    return (px,py)

#=======================about label===================
def read_pointlabel(inlabelfile,xscale=1.0,yscale=1.0):
    labelsfile=open(inlabelfile,'r')
    vs_filespoints=[]        
    for eachline in labelsfile:
        data=eachline.strip().split(' ')
        filename=data[0]
        vspoints=[]
        num=(len(data)-1)/2
        for i in range(int(num)):   
            px=round(int(data[1+i*2])/xscale)
            py=round(int(data[1+i*2+1])/yscale)
            vspoints.append((px,py))
        image_info={"filename":filename,"vspoints":vspoints}
        vs_filespoints.append(image_info)
    return vs_filespoints

#src: 2760*3680,path have black space
def read_bigboxlabel_to_list(inlabelfile):
    vs_boxlabels=[]
    fileobj=open(inlabelfile,'r')
    for eachline in fileobj:
        data=eachline.strip().split(' ')
        if len(data)<3:
            print("Error length labels",len(data))
            break
        filename=data[0]+" "+data[1]
        vsrects=[]
        num=(len(data)-3)/4
        vsrects=[]
        for i in range(int(num)):   
            x=int(data[3+i*4])
            y=int(data[3+i*4+1])
            width=int(data[3+i*4+2])
            height=int(data[3+i*4+3])
            rect=(x,y,width,height)
            vsrects.append(rect)
        image_info={"filename":filename,"vsrects":vsrects}
        #print(image_info["filename"],image_info["vsrects"])
        vs_boxlabels.append(image_info)
    return vs_boxlabels

def read_boxlabel(inlabelfile,xscale=1.0,yscale=1.0):
    vs_boxlabels=[]
    fileobj=open(inlabelfile,'r')
    for eachline in fileobj:
        data=eachline.strip().split(' ')
        if len(data)<3:
            print("label length is too short!",len(data))
            break
        num=(len(data)-2)/4
        if num !=int(data[1]):
            print("label length is not right!",num,int(data[1]))
            break
        filename=data[0]
        vs_rects=[]
        for i in range(int(num)):   
            x=round(int(data[2+i*4])/xscale)
            y=round(int(data[2+i*4+1])/yscale)
            width=round(int(data[2+i*4+2])/xscale)
            height=round(int(data[2+i*4+3])/yscale)
            rect=(x,y,width,height)
            vs_rects.append(rect)
        image_info={"filename":filename,"vsrects":vs_rects}
        #print(image_info["filename"],image_info["vsrects"])
        vs_boxlabels.append(image_info)
    fileobj.close()
    return vs_boxlabels

def create_norml_box_txt(outlabelfile,vs_boxlabels):
    listText = open(outlabelfile,'w')
    count=0
    for i,boxlabels in enumerate(vs_boxlabels):
        txtline=boxlabels["filename"][:]
        vsboxlabels=boxlabels["vsrects"]
        if len(vsboxlabels)==0:
            continue
        else:
            txtline += " "+ str(len(vsboxlabels))
            for boxlabels in vsboxlabels:
                txtline += " "+ str(boxlabels[0]) + " " + str(boxlabels[1])+" "+ str(boxlabels[2]) + " " + str(boxlabels[3])
            print(i,txtline)
            txtline += '\n'
        count+=1
        listText.write(txtline)
        listText.flush()
    listText.close()

#negative to 0, positive to 1
def dirfile_to_clslabel(indirpath,outlabelfile):
    indirs = os.listdir(indirpath)
    listText = open(outlabelfile,'w')
    txtline=""
    for dir in indirs:
        if "positive" in dir:
            label="1"
        else:
            label="0"
        files=os.listdir(indirpath+dir)
        for eachfile in files:
            txtline += eachfile+' '+label+'\n'

    listText.write(txtline)  

def read_clslabel_to_dic(inlabelfile):
    files=open(inlabelfile,'r') 
    dic_fileclslables={}
    for eachline in files:
        data=eachline.strip().split(' ')
        filename=data[0]
        label=int(data[1])
        dic_fileclslables[filename]=label
    return dic_fileclslables

def get_boundingbox(point,boxsize,img_width,img_height):
    x_start=0
    y_start=0
    box_width=boxsize
    box_height=boxsize

    if( point[0] +box_width/2>img_width):
        x_start=img_width-box_width
    elif( point[0]-box_width/2<0):
        x_start=0
    else:
        x_start=point[0]-box_width/2

    if( point[1] +box_height/2>img_height):
        y_start=img_height-box_height
    elif(point[1]-box_height/2<0):
        y_start=0
    else:
        y_start=point[1]-box_height/2
    
    return int(x_start),int(y_start),int(x_start+box_width),int(y_start+box_height)

def resize_all_images(inpath,outpath,img_size):
    files=get_path_files(inpath)
    print(len(files))
    for i,eachfile in enumerate(files):
        im=cv2.imread(os.path.join(inpath,eachfile))
        if im.any()==None:
           print("imread failed")
        im=cv2.resize(im,img_size)
        cv2.imwrite(os.path.join(outpath,eachfile),im)
        print(i,eachfile)

def read_samefile_map(inmapfile):
    vs_maps=[]
    fileobj=open(inmapfile,'r')
    for eachline in fileobj:
        data=eachline.strip().split(' ')
        if(len(data)!=2):
            print("map file have error content")
            break
        dic_file={"filename1":data[0],"filename2":data[1]}
        vs_maps.append(dic_file)
    return vs_maps
