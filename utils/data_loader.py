import numpy as np
import cv2
import os
import pandas as pd
import shutil
import random
import yaml
from easydict import EasyDict as edict

def read_cfg(filename):
    f=open(filename)
    img_cfg=edict(yaml.load(f))
    return img_cfg

class ImageDataLoader():
    def __init__(self, base_path, img_path, gtdes_path, btrain=False):
        self.des_scale=4
        self.data_files = [filename for filename in os.listdir(img_path) \
                           if os.path.isfile(os.path.join(img_path,filename))]
        self.data_files.sort()
        img_cfg=read_cfg(os.path.join(base_path,"img_cfg.yml"))
        self.gmean_r=float(img_cfg['mean'][0])
        self.gmean_g=float(img_cfg['mean'][1])
        self.gmean_b=float(img_cfg['mean'][2])
        img_size = (int(img_cfg['size'][0]),int(img_cfg['size'][1]))
        print("mean:",self.gmean_r,self.gmean_g,self.gmean_b,"size:",img_size)

        self.btrain = btrain
        self.img_ht=img_size[0]
        self.img_wd=img_size[1]
        self.samples_num = len(self.data_files)
        self.blob_img = np.zeros((self.samples_num, img_size[0], img_size[1], 3), dtype='float32')
        self.blob_gtdes = np.zeros((self.samples_num, round(img_size[0]/self.des_scale), round(img_size[1]/self.des_scale), 1), dtype='float32')
        self.blob_gtcls = np.zeros((self.samples_num,), dtype='uint8')

        print ('Pre-loading the data. This may take a while...')

        for idx,fname in enumerate(self.data_files):
            img = cv2.imread(os.path.join(img_path,fname))#,0)
            if img.shape[1]!=img_size[1] or img.shape[0]!=img_size[0]:
                img = cv2.resize(img,(img_size[1],img_size[0]))
            img = img.astype(np.float32, copy=False)
            ht = img.shape[0]
            wd = img.shape[1]
            ht_1 = (ht/self.des_scale)*self.des_scale
            wd_1 = (wd/self.des_scale)*self.des_scale
            img = cv2.resize(img,(int(wd_1),int(ht_1)))
            self.blob_img[idx,:,:,:] = img
            den=None
            if(gtdes_path!=""):
                csv_filename=os.path.join(gtdes_path,os.path.splitext(fname)[0] + '.csv')
                if os.path.exists(csv_filename):
                    den = pd.read_csv(csv_filename, sep=',',header=None).as_matrix()       
                    den = den.astype(np.float32, copy=False)

                    wd_1 = round(wd_1/self.des_scale)
                    ht_1 = round(ht_1/self.des_scale)
                    den = cv2.resize(den,(wd_1,ht_1),cv2.INTER_CUBIC)
                    den = den.reshape((den.shape[0],den.shape[1],1))
                    self.blob_gtdes[idx,:,:,:] = den
            
            if den is None:
                gt_cls=0
            else:
                gt_cls=np.max(den)>0
            self.blob_gtcls[idx] = np.expand_dims(gt_cls, 0) #value to narray
            if idx % 100 == 0:                    
                 print ('Loaded ', idx, '/', self.samples_num, 'files')  
        print ('Completed Loading ', idx+1, 'files')
   

    def imagenet_mean(self,x):
        t=x.copy()
        t[..., 0] -= self.gmean_b
        t[..., 1] -= self.gmean_g
        t[..., 2] -= self.gmean_r
        t = (t*2.0)/255.0    
        return t

    def random_bright_contrastness(self,x, brightness_range,contrastness_range):
        if len(brightness_range) != 2 or len(contrastness_range) != 2:
            raise ValueError('`brightness_rangs,contrastness_ranges hould be tuple or list of two floats. '
                        'Received arg: ', brightness_range,contrastness_range)

        b = np.random.uniform(brightness_range[0], brightness_range[1])
        c = np.random.uniform(contrastness_range[0], contrastness_range[1])
        x = x*c+(b-1.0)*255
        return x

    def augmentation(self,x):
        brightness_range=[0.9,1.1]
        contrastness_range=[0.9,1.1]
        x = self.random_bright_contrastness(x, brightness_range,contrastness_range)  
        return x

    def random_flig(self):
        list=[True,False]
        flig_flip=random.sample(list,1)      
        return flig_flip[0] 

    
    def get_img_size(self):
        return (self.img_ht,self.img_wd)

