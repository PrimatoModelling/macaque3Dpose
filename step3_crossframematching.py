import os
import os.path as osp
import pickle
import sys
import time
import multicam_toolbox as mct
import os.path  as osp
import cv2
import numpy as np
import matplotlib
import copy
import h5py
import yaml
import math
import matplotlib.pyplot as plt
import itertools
import json
import imgstore
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import argparse
import scipy.io
import networkx as nx
import scipy.interpolate
import scipy.signal

const_mindetcnt1 = 12
const_mindetcnt2 = 6
cid_thr = 0.9

def proc(data_name,result_dir_root,raw_data_dir,config_path,save_vid=False,save_vid_cam=2,rmse_thr = 200.0, vidfile_prefix=''):
    result_dir = result_dir_root + '/'+data_name
    Trk, Cid, T = main_proc(config_path, result_dir, rmse_thr=rmse_thr)
    if save_vid:
        visualize(save_vid_cam, config_path, result_dir, raw_data_dir, T=T, vidfile_prefix=vidfile_prefix)

def main_proc(config_path, result_dir, rmse_thr=200.0):

    camparam = get_camparam(config_path)
    n_kp = 17
    n_animal = 4
    fps = 24
    wsize = fps*5

    # get tracklets
    print('get tracklet...')
    Trk, T, n_frame, n_cam = get_tracklets(config_path, result_dir)

    print('trim tracklet...')
    Trk = trim_tracklets(Trk, T, n_frame, n_kp, camparam, config_path)

    # assign IDs
    print('assign IDs...')
    Trk_cid = count_id_detections(T, Trk, n_frame, n_cam)
    Cid = set_id_for_each_frame_of_tracklets(Trk, Trk_cid, n_frame, wsize)
    Trk, Cid = div_3dtracklet(Trk, Cid)

    # clear short tracklets
    print('some cleaning...')
    Trk = remove_single_cam_tracklets(Trk)
    Trk = remove_short_tracklets(Trk, Cid, min_frames=0)

    # stitch tracklets
    print('stitch...')
    Trk, stitch_info = stitch_tracklets(Trk, Cid, T, n_frame, n_cam, n_kp, config_path, camparam)

    # re-assign IDs to tracklets
    print('assign IDs...')
    Trk_cid = count_id_detections(T, Trk, n_frame, n_cam)
    Cid = set_id_for_each_frame_of_tracklets(Trk, Trk_cid, n_frame, wsize)
    Trk, Cid, stitch_info = div_3dtracklet(Trk, Cid, stitch_info)

    print('clean ID duplication...')
    Trk, Cid = breakdown_stitched_tracklet(Trk, Cid, stitch_info)
    Trk_cid = count_id_detections(T, Trk, n_frame, n_cam) # update
    Trk, Cid, Trk_cid = clean_id_duplication(Trk, Cid, Trk_cid, n_frame, wsize, fps)

    print('assign last one...')
    for i_animal in range(n_animal):
        Trk, Cid, flag_update = assign_lastone(Trk, Cid, T, n_kp, n_animal, config_path, camparam, min_duration=12)
        if not flag_update:
            break

    print('create kp2dfile...')
    create_kp2dfile(config_path, result_dir, T, Trk, Cid)

    with open(result_dir + '/track.pickle', 'wb') as f:
        pickle.dump(Trk, f)
    with open(result_dir + '/collar_id.pickle', 'wb') as f:
        pickle.dump(Cid, f)

    with open(result_dir + '/match_keyframe.pickle', 'rb') as f:
        result_keyframe = pickle.load(f)

    return Trk, Cid, T

def assign_lastone(Trk, Cid, T, n_kp, n_animal, config_path, camparam, min_duration=12):
    
    flag_update = False

    ### check existing assignment
    unassigned = []
    assigned = []
    Intv = {}
    for k in Trk.keys():
        if np.sum(Cid[k]>=0) == 0:
            unassigned.append(k)
        else:
            assigned.append(k)
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]

    ### sort from longer to shorter
    intv_len = []
    for k in unassigned:
        intv_len.append(Intv[k][1]-Intv[k][0])
    intv_len = np.array(intv_len)
    I = np.argsort(intv_len)
    I = I[-1::-1]
    unassigned = np.array(unassigned, dtype=int)
    unassigned = unassigned[I].tolist()

    if len(assigned) == 0 or len(unassigned) == 0:
        return Trk, Cid, flag_update

    n_frame = Trk[assigned[0]].shape[0]
    A = np.zeros([n_frame, n_animal])
    for k in assigned:
        for i_c in range(n_animal):
            A[Intv[k][0]:Intv[k][1],i_c] += Cid[k][Intv[k][0]:Intv[k][1]] == i_c
    A = A>0

    for k in tqdm(unassigned):
        intv = Intv[k]

        if intv[1]-intv[0] <= min_duration:
            continue

        a = A[intv[0]:intv[1],:]
        I1 = np.sum(a, axis=1) == 3
        a2 = np.logical_not(a)
        a2 = a2[I1,:]
        
        cnt = np.sum(a2, axis=0)
        i_max = np.argmax(cnt)
        if np.sum(cnt) == 0:
            p = 0.0
        else:
            p = cnt[i_max] / np.sum(cnt)

        if p > 0.8 and cnt[i_max] >= 3:
            cid = i_max
        else:
            continue

        cog_u = None

        flag_overlap = False

        for k2 in assigned:

            I1 = np.zeros(n_frame, bool)
            I2 = np.zeros(n_frame, bool)
            I1[Intv[k][0]:Intv[k][1]] = True
            I2[Intv[k2][0]:Intv[k2][1]] = True

            n_overlap = np.sum(np.logical_and(I1, I2))

            if n_overlap == 0:
                continue

            if n_overlap > (intv[1]-intv[0])/2:
                thr = 2
            else:
                thr = 12

            if cog_u is None:
                cog_u = calc_3dtrace(Trk[k], T, np.arange(intv[0], intv[1]+1), camparam, config_path, n_kp)

            cog_a = calc_3dtrace(Trk[k2], T, np.arange(intv[0], intv[1]+1), camparam, config_path, n_kp)
            d = np.sum((cog_u - cog_a)**2, axis=1)
            I = np.logical_not(np.isnan(d))
            if np.sum(I) >= thr:
                d = d[I]
                rmse = np.sqrt(np.sum(d)/d.shape[0])
                if rmse < 150: 
                    flag_overlap = True
                    break
        
        if flag_overlap:
            continue

        for k2 in assigned:
            cid2 = np.unique(Cid[k2][Intv[k2][0]:Intv[k2][1]])
            cid2 = cid2[cid2>=0]
            if cid2 != cid:
                continue

            I1 = np.zeros(n_frame, bool)
            I2 = np.zeros(n_frame, bool)
            I1[Intv[k][0]:Intv[k][1]] = True
            I2[Intv[k2][0]:Intv[k2][1]] = True

            if np.sum(np.logical_and(I1, I2)) > 0:
                flag_overlap = True
                break
        
        if flag_overlap:
            continue
        
        flag_update = True
        Cid[k][:] = cid
        assigned.append(k)

    return Trk, Cid, flag_update

def breakdown_stitched_tracklet(Trk, Cid, stitch_info):
    
    n_cam = 8

    Intv = {}
    for k in Trk.keys():
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]

    last_key = max(list(Trk.keys()))

    for k in stitch_info.keys():

        if k not in Cid.keys():
            continue

        n_frame = Cid[k].shape[0]

        cid = np.unique(Cid[k][Intv[k][0]:Intv[k][1]])
        cid = np.max(cid)

        frames = stitch_info[k]

        for f in frames:

            trk = -np.ones([n_frame, n_cam], dtype=int)
            trk[f[0]:f[1]+1,:] = Trk[k][f[0]:f[1]+1,:]
            C = -np.ones(n_frame, dtype=int)
            C[f[0]:f[1]+1] = cid
            last_key += 1
            Cid[last_key] = C
            Trk[last_key] = trk

        Trk.pop(k)
        Cid.pop(k)

    return Trk, Cid

def calc_3dpose(kp_2d, config_path,camparam=None):
    n_cam, n_kp, _ = kp_2d.shape

    pos_2d = []
    for i_cam in range(n_cam):
        pos_2d.append(kp_2d[i_cam,:,:2])
    pos_2d_undist = mct.undistortPoints(config_path, pos_2d, omnidir=True, camparam=camparam)

    frame_use = np.ones((n_kp, n_cam), dtype=bool)
    for i_kp in range(n_kp):
        for i_cam in range(n_cam):
            if np.isnan(pos_2d[i_cam][i_kp, 0]):
                frame_use[i_kp, i_cam] = False
            if kp_2d[i_cam,i_kp,2] < 0.3:
                frame_use[i_kp, i_cam] = False

    kp_3d = mct.triangulatePoints(config_path, pos_2d_undist, frame_use, True, camparam=camparam)

    return kp_3d

def calc_3dtrace(trk, T, frames, camparam, config_path, n_kp):

    n_cam = len(T)
    n_frame = len(T[0])
    p3d = np.zeros([n_frame,n_kp,3], dtype=float)
    p3d[:,:,:] = np.nan

    for i_frame in frames:

        if np.sum(trk[i_frame]>=0) < 2:
            continue

        p2d = np.zeros([n_cam, n_kp, 3])
        p2d[:,:,:] = np.nan

        for i_cam in range(n_cam):
            TT = T[i_cam][i_frame]
            for tt in TT:
                bbox_id = tt[0]
                kp = np.array(tt[5])

                if bbox_id == trk[i_frame][i_cam]:
                    p2d[i_cam, :,:] = kp

        p3d[i_frame, :, :] = calc_3dpose(p2d, config_path, camparam)

    trace = np.nanmedian(p3d, axis=1)

    return trace

def calc_dist_pose(p1, p2):
    
    d = np.sum((p1-p2)**2, axis=1)
    d = d[np.logical_not(np.isnan(d))]
    if d.shape[0] == 0:
        rmse = np.nan
    rmse = np.sqrt(np.sum(d)/d.shape[0])

    return rmse

def calc_flow(g):
    
    out_cost = int(1000 * 100)

    nodes = np.unique(g[:,:2])
    nodes = nodes.astype(int)
    n_node = nodes.shape[0]

    best_flow = []
    min_cost = int(1000 * 100 * 1000)
    best_ntrack = -1

    for n_track in range(1, n_node):
        G = nx.DiGraph()

        node_in = []
        for i in nodes:
            node_in.append('IN{:03d}'.format(i))
        
        node_out = []
        for i in nodes:
            node_out.append('OUT{:03d}'.format(i))

        G.add_node("source", demand=-n_track)
        G.add_node("sink", demand=n_track)

        G.add_nodes_from(node_in, demand=1)
        G.add_nodes_from(node_out, demand=-1)
        G.add_edges_from(zip(node_in, node_out), capacity=1, weight=0)
        G.add_edges_from(zip(["source"] * n_node, node_in), capacity=1, weight=out_cost)
        G.add_edges_from(zip(node_out, ["sink"] * n_node), capacity=1, weight=out_cost)

        for i in range(g.shape[0]):
            s = 'OUT{:03d}'.format(int(g[i,0]))
            t = 'IN{:03d}'.format(int(g[i,1]))
            w = int(g[i,2]*100.0) 
            G.add_edge(s, t, weight=w, capacity=1)
        try:
            flowCost, flowDict = nx.capacity_scaling(G)
            
            cnt_in = {}
            cnt_out = {}
            for n in nodes:
                cnt_in[n] = 0
                cnt_out[n] = 0

            for n_in in flowDict.keys():
                fd = flowDict[n_in]
                for k in fd.keys():
                    if 'IN' in k:
                        if fd[k] == 1:
                            cnt_in[int(k[2:])] += 1

            for n in nodes:
                nodename = 'OUT{:03d}'.format(n)
                fd = flowDict[nodename]
                for k in fd.keys():
                    if fd[k] == 1:
                        cnt_out[n] += 1

            if np.sum(np.array(list(cnt_in.values()))>1) > 0:
                continue

            if np.sum(np.array(list(cnt_out.values()))>1) > 0:
                continue

            if flowCost < min_cost:
                min_cost = flowCost
                best_flow = flowDict
                best_ntrack = n_track

        except nx.exception.NetworkXUnfeasible:
            pass

    def reconstruct_path(source, best_flow):
        path = [int(source[3:])]
        for node, flow in best_flow[source].items():
            if flow == 1:
                if node != "sink":
                    path.extend(reconstruct_path(node.replace("IN", "OUT"), best_flow))

        return path

    P = []
    for node, flow in best_flow["source"].items():
        if flow == 1:
            path = reconstruct_path(node.replace("IN", "OUT"), best_flow)
            P.append(path)

    return P

def clean_id_duplication(Trk, Cid, Trk_cid, n_frame, wsize, fps):
    
    n_animal = 4

    Intv = {}
    K = []
    for k in Trk.keys():
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]

    Intv_fixed = copy.deepcopy(Intv)
    k_exclude = []
    k_del = []

    for i_sub in range(n_animal):
        
        # search tracklets with same id
        K = []
        for k in Trk.keys():
            cid = np.unique(Cid[k])
            if np.sum(cid==i_sub):
                K.append(k)

        # check overlap
        cnt_overlap = np.zeros(n_frame, int)
        
        for k in K:
            intv = Intv[k]
            cnt_overlap[intv[0]:intv[1]] = cnt_overlap[intv[0]:intv[1]] + 1

        if np.sum(cnt_overlap>1) == 0:
            #print(i_sub, 'no overlap')
            continue

        # find id detection in tracklets
        Cid_confident = {}
        for k in K:
            cid0 = Trk_cid[k]
            cid1 = -np.ones(n_frame, dtype=int)

            for i_frame in range(max(Intv[k][0],int(wsize/2)), min(Intv[k][1], n_frame-int(wsize/2))):
                
                cnt = np.sum(cid0[i_frame-int(wsize/2):i_frame+int(wsize/2),:], axis=0)
                i_max = np.argmax(cnt)
                if np.sum(cnt) == 0:
                    p = 0.0
                else:
                    p = cnt[i_max] / np.sum(cnt)

                if p > 0.8 and cnt[i_max] >= const_mindetcnt2:
                    I = np.argwhere(cid0[i_frame-int(wsize/2):i_frame+int(wsize/2),i_max])
                    if np.min(I) <= int(wsize/2) and np.max(I) >= int(wsize/2):
                        cid1[i_frame] = i_max

            cid1[:Intv[k][0]] = -1
            cid1[Intv[k][1]:] = -1
            Cid_confident[k] = cid1

        ### remove tracklet without id detection ##############

        # sort from short to long
        intv_len = []
        for k in K:
            intv_len.append(Intv[k][1]-Intv[k][0])
        intv_len = np.array(intv_len)
        I = np.argsort(intv_len)
        K = np.array(K, dtype=int)
        K = K[I].tolist()

        # if tracklet k overlap with other and no id detection, exclude it
        for k1 in K:

            e1 = np.zeros(n_frame, int)
            e2 = np.zeros(n_frame, int)
            e1[Intv[k1][0]:Intv[k1][1]] = 1

            for k2 in K:
                if k2 == k1 or k2 in k_exclude:
                    continue
                e2[Intv[k2][0]:Intv[k2][1]] += 1
            
            if np.sum(e1*e2) == 0:  # no overlap with others
                continue

            f1 = np.argwhere(Cid_confident[k1]==i_sub)

            if f1.shape[0] == 0:
                k_exclude.append(k1)

        ### END: remove tracklet without id detection ##############

        ### remove tracklet without unique contribution

        # delete if no unique contribution
        # for example ....
        #
        # a: ------------------------------------------
        # b:              ---------------
        # tracklet 'b' has no unique contribution. 
        #
        # c: -------------------
        # d:              ------------------------------
        # tracklet 'c' and 'd' have unique contribution. 

        for k1 in K:
            if k1 in k_exclude:
                continue

            e1 = np.zeros(n_frame, int)
            e2 = np.zeros(n_frame, int)
            intv1 = Intv[k1]
            e1[intv1[0]:intv1[1]] = 1
            for k2 in K:
                if k2 == k1:
                    continue
                if k2 in k_exclude:
                    continue
                intv2 = Intv[k2]
                e2[intv2[0]:intv2[1]] = 1

            if np.sum(e1>e2) == 0:
                if np.sum(cnt_overlap[intv1[0]:intv1[1]]>2) == 0:
                    if intv1[0] == 0 or intv1[1] == n_frame-1:
                        pass
                        #print(i_sub, 'overlap at edge')
                    else:
                        k_exclude.append(k1)
                        k_del.append(k1)
                else:
                    k_exclude.append(k1)
                    k_del.append(k1)

        #### END

        # update K
        K2 = []
        for k in K:
            if k not in k_exclude:
                K2.append(k)
        K = K2

        #### shorten or delete the tracklets when overlap

        # sort by intv[0] then intv[1]
        intv2 = []
        for k in K:
            intv2.append(Intv[k])
        intv2 = np.array(intv2)
        I = np.lexsort([intv2[:,1],intv2[:,0]])
        K = np.array(K, dtype=int)
        K = K[I].tolist()

        # shorten or delete the tracklets when overlap, check from start to end
        for i_k in range(len(K)-1):
            k1 = K[i_k]
            k2 = K[i_k+1]
            if k1 in k_exclude:
                continue

            if Intv[k1][1] < Intv[k2][0]:   # no overlap
                continue

            f1 = np.argwhere(Cid_confident[k1]==i_sub).ravel()
            f2 = np.argwhere(Cid_confident[k2]==i_sub).ravel()

            if f1.shape[0] == 0:
                #print('intv:', k1, i_sub, Intv[k1])
                k_exclude.append(k1)
                continue

            if f2.shape[0] == 0:
                #print('intv:', k2, i_sub, Intv[k2])
                k_exclude.append(k2)
                continue

            f1 = np.max(f1)
            f2 = np.min(f2)

            if f1 < f2:
                
                Intv_fixed[k1][1] = f1
                Intv_fixed[k2][0] = f2
                Intv[k1] = Intv_fixed[k1]
                Intv[k2] = Intv_fixed[k2]
                Cid_confident[k1][f1:] = -1
                Cid_confident[k2][:f2] = -1

                #print('fixed', k1, k2, [f1, f2])
                
            else:

                if f2-Intv[k1][0] >= fps and Intv[k2][1]-f1 >= fps:

                    Intv_fixed[k1][1] = f2
                    Intv_fixed[k2][0] = f1
                    Intv[k1] = Intv_fixed[k1]
                    Intv[k2] = Intv_fixed[k2]
                    Cid_confident[k1][f2:] = -1
                    Cid_confident[k2][:f1] = -1

                    #print(i_sub, k1,k2, Intv_fixed[k1], Intv_fixed[k2])

                else:
                    if Intv[k1][1]-Intv[k1][0] > Intv[k2][1]-Intv[k2][0]:
                        k_exclude.append(k2)
                        k_del.append(k2)   
                    else:
                        k_exclude.append(k1)
                        k_del.append(k1)    

        ### END

    ### finalize

    for k in k_exclude:
        Cid[k][:] = -1
    
    for k in Intv_fixed.keys():
        Trk[k][:Intv_fixed[k][0],:] = -1
        Trk[k][Intv_fixed[k][1]:,:] = -1

    for k in Trk.keys():
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0).ravel()
        if I.shape[0] == 0:
            k_del.append(k) # clean empty

    k_del = list(set(k_del))    # remove duplicated keys

    for k in k_del:
        Trk.pop(k)
        Cid.pop(k)
        Trk_cid.pop(k)

    return Trk, Cid, Trk_cid

def clean_kp(kp, ignore_score=False, show_as_possible=True):

    cnt = 0
    for i_kp in range(len(kp)):
        if kp[i_kp][2] > 0.3:
            cnt += 1

    for i_kp in range(len(kp)):

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

def connect_keyframe(config_path, result_dir, divide_2dtrack=True):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    T = []
    for i_cam, id in enumerate(ID):
        with open(result_dir + '/' + str(id) + '/alldata.json', 'r') as f:
            data = json.load(f)
        T.append(data)

    n_cam = 8
    n_kp = 17
    n_frame = len(T[0])

    with open(result_dir + '/match_keyframe.pickle', 'rb') as f:
        result_keyframe = pickle.load(f)

    n_keyframe = len(result_keyframe)

    C = []

    def calc_bbox_similarity(bbox1, bbox2):
        score = np.zeros([len(bbox1), len(bbox2)], float)
        for i in range(len(bbox1)):
            for j in range(len(bbox2)):
                b1 = bbox1[i]
                b2 = bbox2[j]
                a = b1==b2
                a = np.logical_and(a, b1>=0)
                a = np.logical_and(a, b2>=0)
                score[i,j] = np.sum(a)

        return score


    bbox_id_to_change = {}
    for i_cam in range(n_cam):
        bbox_id_to_change[i_cam] = []

    for i_kf in tqdm(range(1, n_keyframe)):
        i_frame_pre = result_keyframe[i_kf-1]['frame']
        i_frame_crnt = result_keyframe[i_kf]['frame']
        bbox_pre = result_keyframe[i_kf-1]['bcomb']
        bbox_crnt = result_keyframe[i_kf]['bcomb']

        n_person_pre = len(bbox_pre)

        P2d_pre = np.zeros([n_person_pre, n_cam, n_kp, 3])
        P2d_pre[:,:,:,:] = np.nan

        bboxsim_score = calc_bbox_similarity(bbox_pre, bbox_crnt)

        row_ind, col_ind = linear_sum_assignment(-bboxsim_score)

        c = []
        for i in range(len(row_ind)):
            if bboxsim_score[row_ind[i], col_ind[i]] > 0:
                c.append([row_ind[i], col_ind[i]])
        
        C.append(c)

        c = np.array(c)
        c = np.reshape(c, [-1, 2])
        for i_cam in range(n_cam):
            bb_pre = []
            bb_crnt = []
            for pid, bb in enumerate(bbox_pre):
                bb_pre.append([bb[i_cam],pid])
            for pid, bb in enumerate(bbox_crnt):
                bb_crnt.append([bb[i_cam],pid])

            for bb1 in bb_pre:
                if bb1[0] < 0:
                    continue
                for bb2 in bb_crnt:
                    if bb2[0] < 0:
                        continue
                    I1 = np.argwhere(c[:,0]==bb1[1]).ravel()
                    I2 = np.argwhere(c[:,1]==bb2[1]).ravel()
                    if I1.shape[0] > 0 and I2.shape[0] > 0:
                        if I2 == I1:
                            if bb1[0] != bb2[0]:
                                #print(i_kf, 'same animal used different box!', I1,I2, bb1[0],bb2[0]) 
                                bbox_id_to_change[i_cam].append([bb1[0], i_frame_pre, i_frame_crnt])
                                bbox_id_to_change[i_cam].append([bb2[0], i_frame_pre, i_frame_crnt])
                        else:
                            if bb1[0] == bb2[0]:
                                #print(i_kf, 'boxes crossed person!', I1,I2, bb1[0],bb2[0]) 
                                bbox_id_to_change[i_cam].append([bb1[0], i_frame_pre, i_frame_crnt])
    
    
    for i_cam in range(n_cam):
        b = np.array(bbox_id_to_change[i_cam])
        b2 = np.unique(b, axis=0)
        bbox_id_to_change[i_cam] = b2.tolist()
        #print(i_cam, b2.tolist())


    ###### Separate bboxes at inconsistent ########

    # find last 2d bbox track id
    n_frame = len(T[0])
    last_bbox_id = -1

    for i_frame in range(n_frame):
        for i_cam in range(n_cam):
            TT = T[i_cam][i_frame]
            for tt in TT:
                bbox_id = tt[0]
                if last_bbox_id < bbox_id:
                    last_bbox_id = bbox_id

    last_bbox_id += 100
    #print(last_bbox_id)

    # give new ids
    T2 = copy.deepcopy(T)
    result_keyframe2 = copy.deepcopy(result_keyframe)
    for i_cam in range(n_cam):

        bc = bbox_id_to_change[i_cam]
        bc = np.array(bc)
        bc = np.reshape(bc, [-1, 3])
        
        I_box = np.unique(bc[:,0])
        for i_box in I_box:
            frames = bc[bc[:,0] == i_box,1:3]
            ids_T = np.ones(n_frame, int) * i_box
            ids_kf = np.ones(n_frame, int) * i_box
            for i_f in range(frames.shape[0]):
                f = frames[i_f,:]
                ids_kf[f[0]+1:f[1]] = -1
                ids_kf[f[1]:] = last_bbox_id
                ids_T[f[0]+1:f[1]] = -10
                ids_T[f[1]:] = last_bbox_id
                last_bbox_id += 1
            
            # for T
            for i_frame in range(n_frame):
                TT = T[i_cam][i_frame]
                for i_tt, tt in enumerate(TT):
                    box_id = tt[0]
                    if box_id == i_box:
                        T2[i_cam][i_frame][i_tt][0] = ids_T[i_frame]

            # for result_keyframe
            for i_kf in range(n_keyframe):
                i_frame = result_keyframe[i_kf]['frame']
                bbox = result_keyframe[i_kf]['bcomb']
                n_person = len(bbox)

                for i_person in range(n_person):
                    if bbox[i_person][i_cam] == i_box:
                        result_keyframe2[i_kf]['bcomb'][i_person][i_cam] = ids_kf[i_frame]

    #################################

    with open(result_dir + '/keyframe_connection.pickle', 'wb') as f:
        pickle.dump(C, f)

    if divide_2dtrack:

        return T2, result_keyframe2, C

    else:

        return T, result_keyframe, C

def count_id_detections(T, Trk, n_frame, n_cam):
    ### get cid info for each frames in each trace
    classnames = ['B', 'd','G', 'R','unknown', 'W'] #macaque
    ####classnames = ['B', 'G', 'R', 'Y'] #marmo
    n_class = len(classnames)

    conf_thr = cid_thr #0.8

    Trk_cid = {}

    for k in tqdm(Trk.keys()):
        trk = Trk[k]
        I = np.argwhere(np.sum(trk>=0, axis=1)>0)
        intv = [np.min(I), np.max(I)]

        t_cid = np.zeros([n_frame, n_class], dtype=int)

        for i_cam in range(n_cam):
            boxid = trk[:,i_cam]

            for i_frame in range(intv[0], intv[1]+1):
                TT = T[i_cam][i_frame]
                for tt in TT:
                    if boxid[i_frame] == tt[0]:
                        cid = tt[6:]
                        if cid[1] > conf_thr:
                            t_cid[i_frame, int(cid[0])] += 1

        Trk_cid[k] = t_cid[:,[0,2,3,5]] #macaque
        #Trk_cid[k] = t_cid #marmo

    return Trk_cid

def create_kp2dfile(config_path, result_dir, T, Trk, Cid):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    n_frame = Trk[list(Trk.keys())[0]].shape[0]

    n_animal = 4
    n_cam = 8
    n_kp = 17

    kp2d = np.zeros([n_animal, n_frame, n_cam, n_kp, 3])
    is_done = np.zeros([n_animal, n_frame, n_cam])

    for i_frame in tqdm(range(n_frame)):

        for k in Trk.keys():

            i_animal = Cid[k][i_frame]

            if i_animal < 0:
                continue

            trk = Trk[k][i_frame,:]
            if np.sum(trk>=0) == 0:
                continue

            for i_cam in range(n_cam):
                if is_done[i_animal, i_frame, i_cam]:
                    continue

                TT = T[i_cam][i_frame]
                for tt in TT:
                    bbox_id = tt[0]
                    kp = np.array(tt[5])

                    if bbox_id == trk[i_cam]:
                        
                        kp2d[i_animal, i_frame, i_cam, :, :] = kp
                        is_done[i_animal, i_frame, i_cam] = True
        

    with open(result_dir + '/kp2d.pickle', 'wb') as f:
        pickle.dump(kp2d, f)
  
def div_3dtracklet(Trk, Cid, stitch_info=None):

    n_cam = 8

    unassigned = []
    assigned = []
    Intv = {}
    for k in Trk.keys():

        if np.sum(Cid[k]>=0) == 0:
            unassigned.append(k)
        else:
            assigned.append(k)
            
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]

    last_key = max(list(Trk.keys()))

    for k in assigned:

        intv = Intv[k]
        cid = np.unique(Cid[k][intv[0]:intv[1]])

        if cid.shape[0] > 1:

            #print(k, intv)
            n_frame = Cid[k].shape[0]

            for cid2 in cid:
                
                A = np.zeros(n_frame, dtype=bool)
                A[intv[0]:intv[1]] = True
                I = to_intv(np.logical_and(Cid[k]==cid2, A))
                #print('intvs:', I, cid2)

                for i in I:
                    C = -np.ones(n_frame, dtype=int)
                    C[i[0]:i[1]+1] = cid2
                    trk = -np.ones([n_frame, n_cam], dtype=int)
                    trk[i[0]:i[1]+1,:] = Trk[k][i[0]:i[1]+1,:]
                    last_key += 1
                    Cid[last_key] = C
                    Trk[last_key] = trk
                    if stitch_info is not None:
                        if k in stitch_info.keys():
                            frames = stitch_info[k]
                            frames2 = []
                            for f in frames:
                                I1 = np.zeros(n_frame, bool)
                                I2 = np.zeros(n_frame, bool)
                                I1[i[0]:i[1]+1] = True
                                I2[f[0]:f[1]+1] = True
                                n_overlap = np.sum(np.logical_and(I1, I2))
                                if n_overlap > 0:
                                    frames2.append(f)
                            stitch_info[last_key] = frames2
                            #print(k,i,stitch_info[k],stitch_info[last_key])


            Trk.pop(k)
            Cid.pop(k)

    if stitch_info is None:
        return Trk, Cid
    else:
        return Trk, Cid, stitch_info

def draw_kps(img, kp, mrksize, clr=None):
    
    cm = plt.get_cmap('hsv', 36)

    kp_con = [
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

def ellipse_line(img, x1, x2, mrksize, clr):
    if x2[0] - x1[0] == 0:
        ang = 90
    else:
        ang = math.atan((x2[1] - x1[1])/(x2[0] - x1[0])) / math.pi * 180
    cen = ((x1[0] + x2[0])/2, (x1[1] + x2[1])/2)
    d = math.sqrt(math.pow(x2[0] - x1[0], 2) + math.pow(x2[1] - x1[1], 2) )
    
    cv2.ellipse(img, (cen, (d, mrksize), ang), clr, thickness=-1)
    return 0

def get_camparam(config_path):
    camparam = {}

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    camparam['camera_id'] = ID

    path_extrin = os.path.dirname(config_path) + '/cam_extrinsic_optim.h5'

    K = []
    xi = []
    D = []
    rvecs = []
    tvecs = []
    for i_cam, id in enumerate(ID):
        with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='r') as f_intrin:
            K.append(f_intrin['/'+str(id)+'/K'][()])
            xi.append(f_intrin['/'+str(id)+'/xi'][()])
            D.append(f_intrin['/'+str(id)+'/D'][()])
        
        with h5py.File(path_extrin, mode='r') as f_extrin:
            rvecs.append(f_extrin['/'+str(id)+'/rvec'][()])
            tvecs.append(f_extrin['/'+str(id)+'/tvec'][()])

    camparam['K'] = K
    camparam['xi'] = xi
    camparam['D'] = D
    camparam['rvecs'] = rvecs
    camparam['tvecs'] = tvecs


    pmat = []
    for i_cam, id in enumerate(ID):
        with h5py.File(path_extrin, mode='r') as f_extrin:
            rvecs = f_extrin['/'+str(id)+'/rvec'][()]
            tvecs = f_extrin['/'+str(id)+'/tvec'][()]
            rmtx, jcb = cv2.Rodrigues(rvecs)
            R = np.hstack([rmtx, tvecs])
            pmat.append(R)

    camparam['pmat'] = pmat

    return camparam

def get_graph(Trk, Cid, T, n_frame, n_cam, n_kp, config_path, camparam):
    
    Intv = {}
    for k in Trk.keys():
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>1)
        Intv[k] = [np.min(I), np.max(I)]

    G = []
    for k1 in Trk.keys():
        for k2 in Trk.keys():
            if k1==k2:
                continue

            # check if box tracking continues
            intv = Intv[k1]
            t_e = Trk[k1][intv[1],:]
            t_e[t_e==-1] = -2
            chk_e = np.sum(Trk[k2][intv[1]:min(intv[1]+120,n_frame)] == t_e, axis=0)
            if np.sum(chk_e>1) == 0:
                continue
            
            # check if overlap is too much
            intv2 = Intv[k2]
            I1 = np.zeros(n_frame, bool)
            I2 = np.zeros(n_frame, bool)
            I1[intv[0]:intv[1]] = True
            I2[intv2[0]:intv2[1]] = True
            n1 = np.sum(I1)
            n2 = np.sum(I2)
            n12 = np.sum(np.logical_and(I1,I2))
            #print(n1,n2,n12, n12/n1, n12/n2)

            if n12/n1 > 0.5 or n12/n2 > 0.5:
                continue

            # calc distance for weight
            def calc_p3d(T, trk, i_frame):
                p2d = np.zeros([n_cam, n_kp, 3])
                p2d[:,:,:] = np.nan
                for i_cam in range(n_cam):
                    TT = T[i_cam][i_frame]
                    for tt in TT:
                        bbox_id = tt[0]
                        kp = np.array(tt[5])

                        if bbox_id == trk[i_frame][i_cam]:
                            p2d[i_cam, :,:] = kp

                p3d = calc_3dpose(p2d, config_path, camparam)
                p3d = np.nanmean(p3d, axis=0)

                return p3d

            i_frame1 = intv[1]
            p3d_1 = calc_p3d(T, Trk[k1], i_frame1)
            #print('s', i_frame1, p3d_1)
            
            I = np.argwhere(np.sum(Trk[k2]>=0, axis=1)>1)
            I = I[I>=intv[1]]
            if I.shape[0] == 0:
                continue
            i_frame2 = I[0]
            p3d_2 = calc_p3d(T, Trk[k2], i_frame2)
            #print('t', i_frame2, p3d_2)
            d = np.sqrt(np.sum(p3d_1 - p3d_2)**2)

            # check if id is different
            if Cid[k1][i_frame1] != -1 and Cid[k2][i_frame2] != -1 and Cid[k1][i_frame1] != Cid[k2][i_frame2]:
                #print(k1,k2, Cid[k1][i_frame1], Cid[k2][i_frame2])
                continue

            # check if id is same
            if Cid[k1][i_frame1] != -1 and Cid[k1][i_frame1] == Cid[k2][i_frame2]:
                d = d * 0.01

            if np.isnan(d):
                continue
            
            G.append([k1,k2,d])

    G = np.array(G)

    G = np.reshape(G,[-1,3])
    #print(G.shape)

    return G

def get_tracklets(config_path, result_dir):
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    n_cam = len(ID)

    T, result_keyframe, result_keyframe_connection = connect_keyframe(config_path, result_dir, divide_2dtrack=True)

    ### clean double color detection
    n_frame = len(T[0])
    for i_cam in range(n_cam):
        for i_frame in range(n_frame):
            TT = T[i_cam][i_frame]
            cnt = np.zeros(20,int)
            for tt in TT:
                bbox_id = tt[0]
                if tt[6] in [0,2,3,5] and tt[7] > cid_thr:
                    cnt[tt[6]] += 1
            
            I = np.argwhere(cnt>1)
            for i_det in I:
                for i_box, tt in enumerate(TT):
                    if tt[6] == i_det:
                        T[i_cam][i_frame][i_box][7] = 0.0

    ### merge connected traces
    n_kf = len(result_keyframe)
    n_frame = result_keyframe[-1]['frame']

    crnt_ids = np.arange(len(result_keyframe[0]['bcomb']), dtype=int)

    if len(result_keyframe[0]['bcomb']) == 0:
        cnt = 0
    else:
        cnt = max(crnt_ids)+1

    Trk = {}
    for i_kf in tqdm(range(1,n_kf)):
        f_pre = result_keyframe[i_kf-1]['frame']
        f_crnt = result_keyframe[i_kf]['frame']

        pre_ids = copy.deepcopy(crnt_ids)

        c = result_keyframe_connection[i_kf-1]

        for i_box, pid in enumerate(pre_ids):
            if pid not in Trk.keys():
                Trk[pid] = -np.ones([n_frame, n_cam], dtype=int)
                
            flag_connection = False
            for i_c in range(len(c)):
                if i_box == c[i_c][0]:
                    bbox_pre = result_keyframe[i_kf-1]['bcomb'][c[i_c][0]]
                    bbox_crnt = result_keyframe[i_kf]['bcomb'][c[i_c][1]]
                    
                    a1 = (bbox_pre >= 0)
                    a2 = (bbox_crnt >= 0)
                    a3 = np.logical_and(a1,a2)
                    a3 = np.logical_not(np.logical_and(a3, bbox_pre != bbox_crnt))
                    a1 = np.logical_and(a1,a3)
                    a2 = np.logical_and(a2,a3)

                    bbox_to_use = -np.ones(n_cam, dtype=int)
                    bbox_to_use[a2] = bbox_crnt[a2]
                    bbox_to_use[a1] = bbox_pre[a1]  # prev keyframe has priority

                    flag_connection = True
                    for i_cam in range(n_cam):
                        Trk[pid][f_pre:f_crnt,i_cam] = bbox_to_use[i_cam]

            if not flag_connection:
                pass


        crnt_ids = -np.ones(len(result_keyframe[i_kf]['bcomb']), dtype=int)
        for i_c in range(len(c)):
            crnt_ids[c[i_c][1]] = pre_ids[c[i_c][0]]

        for i_ids in range(len(crnt_ids)):
            if crnt_ids[i_ids] < 0:
                crnt_ids[i_ids] = cnt
                cnt += 1

    # clean no track
    K = []
    for k in Trk.keys():
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        if I.shape[0] == 0:
            K.append(k)
    for k in K:
        Trk.pop(k)

    return Trk, T, n_frame, n_cam

def interp_pos(x, n_max_frame_intv=5):

    mask_a = np.logical_not(np.isnan(x))
    I = to_intv(np.logical_not(mask_a))
    II = (I[:,1]-I[:,0]) > n_max_frame_intv
    mask_b = I[II,:]

    t0 = np.arange(x.shape[0])
    xx = x[mask_a]
    tt = t0[mask_a]

    f = scipy.interpolate.interp1d(tt, xx, fill_value='extrapolate')
    x2 = f(t0)
    x2 = scipy.signal.medfilt(x2,5) # median filter
    for i in range(mask_b.shape[0]):
        x2[mask_b[i,0]:mask_b[i,1]] = np.nan

    return x2

def remove_short_tracklets(Trk, Cid, min_frames=24):

    k_del = []
    for k in Trk.keys():
        if np.sum(Cid[k]>=0) == 0:
            I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
            if np.max(I) - np.min(I) <= min_frames:
                k_del.append(k)

    #print('del:', k_del)
    for k in k_del:
        Trk.pop(k)

    return Trk

def remove_single_cam_tracklets(Trk):
    
    k_del = []
    for k in Trk.keys():
        a = np.array(Trk[k])
        a = a>=0
        np.sum(np.sum(a, axis=1)>1)
        a = np.sum(a, axis=1)>1
        if np.sum(a) == 0:
            k_del.append(k)

    #print('del:', k_del)
    for k in k_del:
        Trk.pop(k)

    return Trk

def reproject(i_cam, p3d, config_path, camparam=None):

    if camparam is None:

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
    else:
        K = camparam['K'][i_cam]
        xi = camparam['xi'][i_cam]
        D = camparam['D'][i_cam]
        rvecs = camparam['rvecs'][i_cam]
        tvecs = camparam['tvecs'][i_cam]

    pts, _ = cv2.omnidir.projectPoints(np.reshape(p3d, (-1,1,3)), rvecs, tvecs, K, xi[0][0], D)

    return pts[:,0,:]

def set_id_for_each_frame_of_tracklets(Trk, Trk_cid, n_frame, wsize):
    
    Intv = {}
    for k in Trk.keys():

        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]
        
    Cid = {}
    for k in tqdm(Trk_cid.keys()):
        cid0 = Trk_cid[k]

        # initialize labels
        cid1 = -np.ones(n_frame, dtype=int)
        cid2 = -np.ones(n_frame, dtype=int)

        # find the times exceed threshold of ID assignment -> cid1
        for i_frame in range(max(Intv[k][0],int(wsize/2)), min(Intv[k][1], n_frame-int(wsize/2))):
            
            cnt = np.sum(cid0[i_frame-int(wsize/2):i_frame+int(wsize/2),:], axis=0)
            i_max = np.argmax(cnt)
            if np.sum(cnt) == 0:
                p = 0.0
            else:
                p = cnt[i_max] / np.sum(cnt)

            if p > 0.8 and cnt[i_max] >= const_mindetcnt1:
                cid1[i_frame] = i_max

        # assign all frames -> cid2
        uid = np.unique(cid1[Intv[k][0]:Intv[k][1]])

        if np.sum(uid>=0) == 0: # tracklet contains no ID 
            
            # try to check globally
            cnt = np.sum(cid0, axis=0)
            i_max = np.argmax(cnt)
            if np.sum(cnt) == 0:
                p = 0.0
            else:
                p = cnt[i_max] / np.sum(cnt)
            if p > 0.8 and cnt[i_max] >= const_mindetcnt1:
                cid2[:] = i_max

        elif np.sum(uid>=0) == 1: # tracklet contains single ID 
            cid2[:] = uid[uid>=0]
            
        else:   # tracklet contains >= 2 IDs
            pre_id = -1
            pre_frame = 0
            
            for i_frame in range(n_frame):
                crnt_id = cid1[i_frame]

                if crnt_id >= 0:
                    if crnt_id != pre_id:

                        if pre_id == -1:
                            #fill beginning
                            cid2[0:i_frame] = crnt_id
                        else:
                            if i_frame - pre_frame > 1:
                                
                                # find midpoint
                                chk_intv = [max(1,pre_frame-int(wsize/2)), i_frame]
                                I_det_preid = np.argwhere(cid0[:,pre_id] > 0)
                                I_det_preid = I_det_preid[np.logical_and(I_det_preid >= chk_intv[0], I_det_preid <= chk_intv[1])]
                                if I_det_preid.shape[0] > 0:
                                    I_det_preid = max(I_det_preid)
                                else:
                                    I_det_preid = pre_frame

                                chk_intv = [pre_frame, min(i_frame+int(wsize/2), n_frame)]
                                I_det_crntid = np.argwhere(cid0[:,crnt_id] > 0)
                                I_det_crntid = I_det_crntid[np.logical_and(I_det_crntid >= chk_intv[0], I_det_crntid <= chk_intv[1])]
                                if I_det_crntid.shape[0] > 0:
                                    I_det_crntid = min(I_det_crntid)
                                else: 
                                    I_det_crntid = i_frame

                                if I_det_preid < I_det_crntid:
                                    midpoint = int((I_det_crntid-I_det_preid)/2)+I_det_preid
                                else:
                                    midpoint = int((i_frame-pre_frame)/2)+pre_frame

                                # separate at midpoint
                                cid2[pre_frame:midpoint] = pre_id
                                cid2[midpoint:i_frame] = crnt_id
                            
                    else:
                        # fill interval between same ID
                        cid2[pre_frame:i_frame] = crnt_id
        
                    pre_id = crnt_id
                    pre_frame = i_frame

            cid2[pre_frame:] = pre_id   # fill the rest

        Cid[k] = cid2

    return Cid

def stitch_tracklets(Trk, Cid, T, n_frame, n_cam, n_kp, config_path, camparam):

    stitch_info = {}

    g = get_graph(Trk, Cid, T, n_frame, n_cam, n_kp, config_path, camparam)

    if g.shape[0] == 0:
        return Trk, stitch_info

    F = calc_flow(g)
    
    Intv = {}
    for k in Trk.keys():
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]

    last_key = max(list(Trk.keys()))
    k_del = []
    for f in F:
        if len(f) > 1:
            trk1 = Trk[f[0]]
            frames = []
            for k in f:
                trk2 = Trk[k]
                I = trk1 == -1
                trk1[I] = trk2[I]
                frames.append(Intv[k])

            last_key += 1
            Trk[last_key] = trk1
            stitch_info[last_key] = frames
            #print(last_key, f)

            k_del.extend(f)

    #print('del:', k_del)
    for k in k_del:
        Trk.pop(k)

    return Trk, stitch_info

def to_intv(I):
    I = np.array(I, dtype=int)

    if I[-1] == 1:
        I = np.append(I, 0)
        
    d = np.diff(np.append(np.array([0]), I))

    start = np.where(d == 1)
    stop = np.where(d == -1)

    start = start[0]
    stop = stop[0]
    intv = np.array([start, stop]).T

    return intv

def trim_tracklets(Trk, T, n_frame, n_kp, camparam, config_path):

    # get tracklet interval
    Intv = {}
    K = []
    for k in Trk.keys():
        I = np.argwhere(np.sum(Trk[k]>=0, axis=1)>0)
        Intv[k] = [np.min(I), np.max(I)]
        K.append(k)

    # sort from short to long
    intv_len = []
    for k in K:
        intv_len.append(Intv[k][1]-Intv[k][0])
    intv_len = np.array(intv_len)
    I = np.argsort(intv_len)
    K = np.array(K, dtype=int)
    K = K[I].tolist()

    # trimming
    Trk2 = copy.deepcopy(Trk)

    for k1 in K:
        for k2 in K:

            if k2 == k1:
                continue

            e1 = np.zeros(n_frame, int)
            e2 = np.zeros(n_frame, int)
            e1[Intv[k1][0]:Intv[k1][1]+1] = 1
            e2[Intv[k2][0]:Intv[k2][1]+1] = 1

            n_overlap = np.sum(e1*e2)

            if n_overlap == 0:  # no overlap with others
                continue

            if n_overlap > np.sum(e1)/3 or n_overlap > np.sum(e2)/3 or n_overlap > 12: # too much overlap
                continue

            case_a = Intv[k1][0] > Intv[k2][0] and Intv[k1][1] > Intv[k2][1]
            case_b = Intv[k2][0] > Intv[k1][0] and Intv[k2][1] > Intv[k1][1]

            if not case_a and not case_b:
                continue

            frames_overlap = np.argwhere(e1*e2 == 1).ravel()

            trace1 = calc_3dtrace(Trk2[k1], T, frames_overlap, camparam, config_path, n_kp)
            trace2 = calc_3dtrace(Trk2[k2], T, frames_overlap, camparam, config_path, n_kp)

            rmse = calc_dist_pose(trace1, trace2)
            
            if rmse < 150:
                if case_a:
                    Intv[k1][0] = Intv[k2][1]+1
                    Trk2[k1][:Intv[k2][1]+1,:] = -1
                elif case_b:
                    Intv[k1][1] = Intv[k2][0]-1 
                    Trk2[k1][Intv[k2][0]:,:] = -1

                #print('trimmed', k1,k2,Intv[k1],Intv[k2],n_overlap,rmse)

    return Trk2

def visualize(vis_cam, config_path, result_dir, raw_data_dir, T=None, vidfile_prefix=''):

    camparam = get_camparam(config_path)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    if T is None:
        T = []
        for i_cam, id in enumerate(ID):
            with open(result_dir + '/' + str(id) + '/alldata.json', 'r') as f:
                data = json.load(f)
            T.append(data)

    data_name = os.path.basename(result_dir)
    data_name = data_name.split('.')[0]
    n_cam = len(ID)
    S = []
    F = []
    for i_cam in range(n_cam):
        mdata = raw_data_dir + '/' + data_name + '.' + str(ID[i_cam]) + '/metadata.yaml'
        frame_num = np.load(result_dir + '/' + str(ID[i_cam]) + '/frame_num.npy')
        store = imgstore.new_for_filename(mdata)
        S.append(store)
        F.append(frame_num)

    with open(result_dir + '/track.pickle', 'rb') as f:
        Trk = pickle.load(f)
    with open(result_dir + '/collar_id.pickle', 'rb') as f:
        Cid = pickle.load(f)

    n_frame = Trk[list(Trk.keys())[0]].shape[0]

    n_kp = 17

    vw = cv2.VideoWriter(vidfile_prefix + '{:d}.mp4'.format(ID[vis_cam]), cv2.VideoWriter_fourcc(*'mp4v'), 24, (800,600))
    
    frame_number = -1

    for i_frame in tqdm(range(0, n_frame, 3)):

        i = F[vis_cam][i_frame]

        if frame_number >= F[vis_cam][i_frame]:
            pass
        if frame_number == -1:
            img, (frame_number, frame_time) = S[vis_cam].get_image(F[vis_cam][i_frame])
        else:
            while frame_number < F[vis_cam][i_frame]:
                img, (frame_number, frame_time) = S[vis_cam].get_next_image()

        img2 = copy.deepcopy(img)

        for k in Trk.keys():
            trk = Trk[k][i_frame,:]
            if np.sum(trk>=0) == 0:
                continue
            p2d = np.zeros([n_cam, n_kp, 3])
            p2d[:,:,:] = np.nan

            for i_cam in range(n_cam):
                TT = T[i_cam][i_frame]
                for tt in TT:
                    bbox_id = tt[0]
                    kp = np.array(tt[5])

                    if bbox_id == trk[i_cam]:
                        p2d[i_cam, :,:] = kp
        
            p3d = calc_3dpose(p2d, config_path,camparam)
                
            clrs = [(255,0,0),(0,255,0),(0,0,255),(255,255,255)]
            if Cid[k][i_frame] >= 0:
                clr = clrs[Cid[k][i_frame]]
            else:
                clr = (0, 0, 0)

            x = p3d
            a = (x[5,:] + x[6,:])/2
            x = np.concatenate([x,a[np.newaxis,:]],axis=0)
            p = reproject(vis_cam, x, config_path, camparam)
            p = np.concatenate([p, np.ones([p.shape[0], 1])], axis=1)
            kp = p.tolist()
            mrksize = 3
            clean_kp(kp)
            draw_kps(img2, kp, mrksize, clr)

            x_min = np.nanmin(p[:,0])
            y_min = np.nanmin(p[:,1])

            if x_min > 3000 or x_min < -1000:
                x_min = np.nan
            if y_min > 3000 or y_min < -1000:
                y_min = np.nan

            if np.isnan(x_min) != True and np.isnan(y_min) != True:
                cv2.putText(img2, str(k), (int(x_min), int(y_min)), #(int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2.0,
                    color=clr, thickness=5)

        #cv2.putText(img2, 'Frame:{:05d}'.format(i_frame), (int(50), int(50)), #(int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)), 
        #    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #    fontScale=1.5,
        #    color=(255,255,255), thickness=3)

        """
        BB = B[vis_cam][i_frame]

        if len(BB) > 0:
            for b in BB:
                cv2.rectangle(img2, (b[0],b[1]), (b[2],b[3]), color=(0,0,0), thickness=3)
        """

        img2 = cv2.resize(img2, [800,600])
        vw.write(img2)

    vw.release()

if __name__ == '__main__':

    pass