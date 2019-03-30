import numpy as np
import cv2
import os
import cv2
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Densemap():
    def __init__(self, image_size=(280,400),heatmap_size=(35,50),sigma=1):
        self.image_size=np.array(image_size)
        self.heatmap_size=np.array(heatmap_size)
        self.sigma=sigma

    # # Generate gaussian target
    def generate_target(self,joints):
        g_num_joints=joints.shape[0]
        if g_num_joints==0:
            return np.zeros((1,self.heatmap_size[0],self.heatmap_size[1]),dtype=np.float32) 
        target = np.zeros((g_num_joints,self.heatmap_size[0],self.heatmap_size[1]),dtype=np.float32) 
        tmp_size = self.sigma * 3
        for joint_id in range(g_num_joints):
            #setp1判断是否越界，并计算区域范围
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[1] or ul[1] >= self.heatmap_size[0] \
                   or br[0] < 0 or br[1] < 0:
                print("point is not right")
                # If not, just return the image as is
                continue

            #step2创建高斯
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            #step3把高斯Copy到target中
            # Usable gaussian range
            if br[0]>=self.heatmap_size[1]:
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[1]) - ul[0]+1
            else:
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[1]) - ul[0]

            if br[1]>=self.heatmap_size[0]:
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[0]) - ul[1]+1
            else:
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[0]) - ul[1]

            # Image range
            if ul[0]-1 <= 0:
                print("repaire x:",ul[0],br[0])
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[1])
            else:
                img_x = max(0, ul[0]-1), min(br[0]-1, self.heatmap_size[1])

            if ul[1]-1 <= 0:
                print("repaire y:",ul[1],br[1])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[0])
            else:
                img_y = max(0, ul[1]-1), min(br[1]-1, self.heatmap_size[0])

            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target

   #combine with max value
    def target_merge(self,target):
        output = np.zeros((self.heatmap_size[1],self.heatmap_size[0]),dtype=np.float32) 
        output=target[0]
        for c,gh in enumerate(target):
            if c==0:
                continue
            rows,cols = gh.shape[0],gh.shape[1]
            for row in range(rows):
                for col in range(cols):
                    if output[row,col]<gh[row,col]:
                        output[row,col]=gh[row,col]
        return output

    def disp3D(self,z):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        wd=z.shape[0]
        ht=z.shape[1]
        x = np.linspace(0, ht, ht)
        y = np.linspace(0, wd, wd)
        x,y=np.meshgrid(x,y)
        print(len(x),len(y),z.shape[0],z.shape[1])
        ax.plot_surface(x, y, z)
        plt.show()

