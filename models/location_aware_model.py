import numpy as np
import cv2
import numpy as np
import os
from os.path import splitext

class PartExtraction():
    def __init__(self,base_path,img_size):
        self.base_path=base_path
        self.img_size=img_size
        self.img=None

    def read_img(self,filepre):
        self.img=cv2.imread(os.path.join(self.base_path,filepre+".JPG"))
        #self.img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.img=cv2.resize(self.img,(1760,1120))

    def get_partimg(self,points):
        ret_imgs = np.zeros((len(points), self.img_size[0], self.img_size[1], 3), dtype='float32')
        for i,point in enumerate(points):
            x1,y1,x2,y2=get_img_bbox(point,self.img_size,self.img.shape[1],self.img.shape[0])
            ret_imgs[i,...]=self.img[y1:y2,x1:x2]
        return ret_imgs


def get_img_bbox(point,boxsize,img_width,img_height):
    x_start=0
    y_start=0
    box_width=boxsize[0]
    box_height=boxsize[1]
    #352*224
    x=(point[0]*img_width)/88
    y=(point[1]*img_height)/56
    if( x + box_width/2 > img_width):
        x_start=img_width-box_width
    elif( x - box_width/2 < 0):
        x_start=0
    else:
        x_start=x-box_width/2

    if( y + box_height/2 > img_height ):
        y_start=img_height-box_height
    elif( y- box_height/2<0 ):
        y_start=0
    else:
        y_start=y-box_height/2
    return int(x_start),int(y_start),int(x_start+box_width),int(y_start+box_height)
    #return int(round(x_start)),int(round(y_start)),int(round(x_start+box_width)),int(round(y_start+box_height))


class Location_aware_model():
    def __init__(self, lcnn_model, patchnet, orgroi_data_path,parm_k=0,parm_t=0):
        self.lcnn_model=lcnn_model
        self.patchnet =patchnet
        self.part_extracter=PartExtraction(orgroi_data_path,self.patchnet.get_size())
        self.parm_k=parm_k
        self.parm_t=parm_t

    def set_superparm(self,parm_k,parm_t):
        self.parm_k=parm_k
        self.parm_t=parm_t
         
    def predict_img(self,pre_des):
        cout=0
        off_set=2
        tmp_clss = np.zeros((self.parm_t),dtype='float32')
        tmp_des=pre_des.copy()
        while(np.max(tmp_des)!=0 and cout<self.parm_t):
            points=[]
            des_max=np.max(tmp_des)
            rcs=np.where(tmp_des==des_max)
            point=(rcs[1][0],rcs[0][0])
            x1,y1,x2,y2=get_img_bbox(point,(11,11),tmp_des.shape[1],tmp_des.shape[0])
            tmp_des[y1+1:y2-1,x1+1:x2-1]=0  #clear the box
            if self.parm_k==1:
                points.append(point)
            elif self.parm_k==2:
                points.append((point[0]-off_set,point[1]-off_set))
                points.append((point[0]+off_set,point[1]+off_set))
            elif self.parm_k==3:
                points.append((point[0]-off_set,point[1]))
                points.append((point[0],point[1]-off_set))
                points.append((point[0]+off_set,point[1]+off_set))
            elif self.parm_k==4:
                points.append((point[0]-off_set,point[1]-off_set))
                points.append((point[0]+off_set,point[1]+off_set))
                points.append((point[0]-off_set,point[1]+off_set))
                points.append((point[0]+off_set,point[1]-off_set))
            elif self.parm_k==5:
                points.append((point[0],point[1]))
                points.append((point[0]-off_set,point[1]-off_set))
                points.append((point[0]+off_set,point[1]+off_set))
                points.append((point[0]-off_set,point[1]+off_set))
                points.append((point[0]+off_set,point[1]-off_set))
            ret_imgs=self.part_extracter.get_partimg(points)
            tmp_clss[cout]=np.mean(self.patchnet.predict(ret_imgs))
            cout+=1
        if True in (tmp_clss>0.5):
            return np.max(tmp_clss)
        else:
            return np.mean(tmp_clss)

    def predict_all(self,in_imgs,org_files):
        samples_num=in_imgs.shape[0]
        pr_dess = self.lcnn_model.predict(in_imgs)
        pr_clss=np.zeros(samples_num,dtype='float32')
        for idx in range(samples_num):
            pr_des= pr_dess[idx]
            pr_des=cv2.GaussianBlur(pr_des,(3,3),0)
            pr_des[pr_des<0.1]=0
            if np.max(pr_des)!=0:
                filepre, fileext = splitext(org_files[idx])
                self.part_extracter.read_img(filepre)
                pr_clss[idx]=self.predict_img(pr_des)

        return pr_dess,pr_clss





