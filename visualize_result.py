import os
from unittest import result
import multicam_toolbox as mct

import copy
import os
import cv2
import numpy as np
from tqdm import tqdm
from math import floor, ceil 
import yaml
import matplotlib.pyplot as plt
import math
import h5py
import pickle
import imgstore
import scipy.io
import glob

def ellipse_line(img, x1, x2, mrksize, clr):
    if x2[0] - x1[0] == 0:
        ang = 90
    else:
        ang = math.atan((x2[1] - x1[1])/(x2[0] - x1[0])) / math.pi * 180
    cen = ((x1[0] + x2[0])/2, (x1[1] + x2[1])/2)
    d = math.sqrt(math.pow(x2[0] - x1[0], 2) + math.pow(x2[1] - x1[1], 2) )
    
    cv2.ellipse(img, (cen, (d, mrksize), ang), clr, thickness=-1)
    return 0

def clean_kp(kp, ignore_score=False, show_as_possible=True):

    cnt = 0
    for i_kp in range(len(kp)):
        if kp[i_kp][2] > 0.3:
            cnt += 1

    for i_kp in range(len(kp)):

        if i_kp == 1 or i_kp == 2:
            kp[i_kp] = None
            continue

        if show_as_possible:
            if cnt == 0:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][0]):
                kp[i_kp] = None
            elif kp[i_kp][0] > 3000 or kp[i_kp][0] < -1000 or kp[i_kp][1] > 3000 or kp[i_kp][1] < -1000:
                kp[i_kp] = None
            else:
                kp[i_kp] = kp[i_kp][0:2]
        else:
            if kp[i_kp][2] < 0.3 and not ignore_score:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][2]) and not ignore_score:
                kp[i_kp] = None
            elif np.isnan(kp[i_kp][0]):
                kp[i_kp] = None
            elif kp[i_kp][0] > 3000 or kp[i_kp][0] < -1000 or kp[i_kp][1] > 3000 or kp[i_kp][1] < -1000:
                kp[i_kp] = None
            else:
                kp[i_kp] = kp[i_kp][0:2]

def add_neckkp(kp):
    if kp[5] is not None and kp[6]is not None:
        d = [(kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2]
    else:
        d = None
    kp.append(d)

def draw_kps(img, kp, mrksize, clr=None):
    
    cm = plt.get_cmap('hsv', 36)

    kp_con = [
            #{'name':'0_2','color':cm(27), 'bodypart':(0,2)},
            #{'name':'0_1','color':cm(31),'bodypart':(0,1)},
            #{'name':'2_4','color':cm(29), 'bodypart':(2,4)},
            #{'name':'1_3','color':cm(33),'bodypart':(1,3)},
            {'name':'0_4','color':cm(29), 'bodypart':(0,4)},
            {'name':'0_3','color':cm(33),'bodypart':(0,3)},
            {'name':'6_8','color':cm(5),'bodypart':(6,8)},
            {'name':'5_7','color':cm(10),'bodypart':(5,7)},
            {'name':'8_10','color':cm(7),'bodypart':(8,10)},
            {'name':'7_9','color':cm(12),'bodypart':(7,9)},
            {'name':'12_14','color':cm(16),'bodypart':(12,14)},
            {'name':'11_13','color':cm(22),'bodypart':(11,13)},
            {'name':'14_16','color':cm(18),'bodypart':(14,16)},
            {'name':'13_15','color':cm(24),'bodypart':(13,15)},
            {'name':'0_17','color':cm(26),'bodypart':(0,17)},
            {'name':'17_6','color':cm(2),'bodypart':(17,6)},
            {'name':'17_5','color':cm(3),'bodypart':(17,5)},
            {'name':'17_12','color':cm(14),'bodypart':(17,12)},
            {'name':'17_11','color':cm(20),'bodypart':(17,11)}
            ]

    kp_clr = [cm(0), cm(32), cm(28), cm(34), cm(30), cm(9), cm(4), cm(11), cm(6), 
              cm(13), cm(8), cm(21), cm(15), cm(23), cm(17), cm(25), cm(19), cm(1)] 

    for i in reversed(range(len(kp))):
        if kp[i] is not None:
            c = kp_clr[i]
            if clr is None:
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), mrksize, (c[0]*255, c[1]*255, c[2]*255), thickness=-1)
            else:
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), mrksize, (clr[0]*255, clr[1]*255, clr[2]*255), thickness=-1)
            #ax.plot(kp[i][0], kp[i][1], color=kp_clr[i], marker='o', ms=8*mrksize, alpha=1, markeredgewidth=0)
    
    for i in reversed(range(len(kp_con))):
        j1 = kp_con[i]['bodypart'][0]
        j2 = kp_con[i]['bodypart'][1]
        c = kp_con[i]['color']
        if kp[j1] is not None and kp[j2] is not None:
            if clr is None:
                ellipse_line(img, kp[j1], kp[j2], mrksize, (c[0]*255, c[1]*255, c[2]*255))
            else:
                ellipse_line(img, kp[j1], kp[j2], mrksize, (clr[0]*255, clr[1]*255, clr[2]*255))

def reproject(config_path, i_cam, p3d):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    id = ID[i_cam]

    path_intrin = os.path.dirname(config_path) + '/cam_intrinsic.h5'
    path_extrin = os.path.dirname(config_path) + '/cam_extrinsic_optim.h5'

    with h5py.File(path_intrin, mode='r') as f_intrin:
        mtx = f_intrin['/'+str(id)+'/mtx'][()]
        dist = f_intrin['/'+str(id)+'/dist'][()]
        K = f_intrin['/'+str(id)+'/K'][()]
        xi = f_intrin['/'+str(id)+'/xi'][()]
        D = f_intrin['/'+str(id)+'/D'][()]
    with h5py.File(path_extrin, mode='r') as f_extrin:
        rvecs = f_extrin['/'+str(id)+'/rvec'][()]
        tvecs = f_extrin['/'+str(id)+'/tvec'][()]

    pts, _ = cv2.omnidir.projectPoints(np.reshape(p3d, (-1,1,3)), rvecs, tvecs, K, xi[0][0], D)

    return pts[:,0,:]

def proc(data_name, i_cam, config_path, raw_data_dir=None):

    result_dir = './results3D/' + data_name

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    vid_path = './output/' + data_name + '_{:d}.mp4'.format(ID[i_cam])

    if os.path.exists(vid_path):
        print('already exists:', vid_path)
        return

    if os.path.exists(result_dir + '/kp3d_fxdJointLen.pickle'):
        kp3d_file = result_dir + '/kp3d_fxdJointLen.pickle'
    elif os.path.exists(result_dir + '/kp3d.pickle'):
        kp3d_file = result_dir + '/kp3d.pickle'
    else:
        print('no kp3d file:', data_name)
        return
        

    print('generating...', vid_path)

    ignore_score = False
    show_as_possible = True

    vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (800, 600))

    with open(kp3d_file, 'rb') as f:
        data = pickle.load(f)

    data_name = os.path.basename(result_dir)
    mdata = raw_data_dir + '/' + data_name.split('.')[0] + '.' + str(ID[i_cam]) + '/metadata.yaml'
    frame_num = np.load(result_dir + '/' + str(ID[i_cam]) + '/frame_num.npy')
    store = imgstore.new_for_filename(mdata)

    X = data['kp3d']
    S = data['kp3d_score']

    n_animal, n_frame, n_kp, _ = X.shape
    n_cam = len(ID)

    clrs = [(1,0,0), (0,1,0), (0,0,1), (1,1,1)]

    frame_number = -1
    for i_frame in tqdm(range(n_frame)):

        if frame_number >= frame_num[i_frame]:
            pass
        if frame_number == -1:
            frame, (frame_number, frame_time) = store.get_image(frame_num[i_frame])
        else:
            while frame_number < frame_num[i_frame]:
                frame, (frame_number, frame_time) = store.get_next_image()

        img = copy.deepcopy(frame)

        for i_animal in range(n_animal):
            x = X[i_animal, i_frame, :, :]
            s = S[i_animal, i_frame, :]

            a = (x[5,:] + x[6,:])/2
            x = np.concatenate([x,a[np.newaxis,:]],axis=0)

            a = (s[5] + s[6])/2
            s = np.concatenate([s,a[np.newaxis]],axis=0)

            p = reproject(config_path, i_cam, x)
            I = np.logical_not(x[:,0]==0)
            I = np.logical_and(I, s>0.0)
            p = np.concatenate([p, I[:,np.newaxis]], axis=1)

            kp = p.tolist()
            mrksize = 3
            clean_kp(kp, ignore_score=ignore_score, show_as_possible=show_as_possible)
            
            if i_animal < 4:
                draw_kps(img, kp, mrksize, clrs[i_animal])
            else: 
                draw_kps(img, kp, mrksize, (0,0,0))

        #cv2.putText(img, 'Frame:{:05d}'.format(i_frame), (int(30), int(50)), 
        #    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #    fontScale=1.5,
        #    color=(255,255,255), thickness=3)

        img = cv2.resize(img, (800,600))

        vw.write(img)

    vw.release()

if __name__ == '__main__':

    pass



    
