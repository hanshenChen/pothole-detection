from models.submodel import create_resnet50
import cv2
import os
import yaml
import numpy as np
from easydict import EasyDict as edict

def printdim(inimg):
    for i in range(inimg.ndim):
        print(inimg.shape[i])

def read_cfg(filename):
    f=open(filename)
    img_cfg=edict(yaml.load(f))
    return img_cfg

class PatchNet():
    def __init__(self,base_path,weightsfile):
        img_cfg=read_cfg(base_path+"/img_cfg.yml")
        self.gmean_r=float(img_cfg['mean'][0])
        self.gmean_g=float(img_cfg['mean'][1])
        self.gmean_b=float(img_cfg['mean'][2])
        self.img_size = (int(img_cfg['size'][0]),int(img_cfg['size'][1]))
        print("mean:",self.gmean_r,self.gmean_g,self.gmean_b,"size:",self.img_size)
        self.model_cls=create_resnet50(self.img_size+((3,)))

        print(weightsfile)
        self.model_cls.load_weights(weightsfile)
     
    def image_mean(self,t,bbgr2rgb=True):
        x=t.copy()
        if bbgr2rgb==True:
            x = x[..., ::-1]
        x[..., 0] -= self.gmean_r#161.09 #R
        x[..., 1] -= self.gmean_g#156.35 #G
        x[..., 2] -= self.gmean_b#132.44 #B
        x = (x*2.0)/255.0   
        return x

    def predict(self,inbatch_img,bbgr2rgb=True):
        batch_img=self.image_mean(inbatch_img,bbgr2rgb)
        et_clss = self.model_cls.predict(batch_img)
        return et_clss

    def get_size(self):
        return self.img_size

  
