import anipose.filter_pose as af 
import numpy as np
import os
import pandas as pd
import glob 
import re

import pickle
import yaml
import toml
import h5py
from aniposelib.cameras import CameraGroup

import argparse

import multicam_toolbox as mct
import argparse

def proj(u, v):
    """Project u onto v"""
    return u * np.dot(v,u) / np.dot(u,u)

def ortho(u, v):
    """Orthagonalize u with respect to v"""
    return u - proj(v, u)

def get_median(all_points_3d, ix):
    pts = all_points_3d[:, ix]
    pts = pts[~np.isnan(pts[:, 0])]
    return np.median(pts, axis=0)

def load_constraints(config, bodyparts, key='constraints'):
    constraints_names = config['triangulation'].get(key, [])
    bp_index = dict(zip(bodyparts, range(len(bodyparts))))
    constraints = []
    for a, b in constraints_names:
        assert a in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(a)
        assert b in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(b)
        con = [bp_index[a], bp_index[b]]
        constraints.append(con)
    return constraints

def correct_coordinate_frame(config, all_points_3d, bodyparts):
    """Given a config and a set of points and bodypart names, this function will rotate the coordinate frame to match the one in config"""
    bp_index = dict(zip(bodyparts, range(len(bodyparts))))
    axes_mapping = dict(zip('xyz', range(3)))

    ref_point = config['triangulation']['reference_point']
    axes_spec = config['triangulation']['axes']
    a_dirx, a_l, a_r = axes_spec[0]
    b_dirx, b_l, b_r = axes_spec[1]

    a_dir = axes_mapping[a_dirx]
    b_dir = axes_mapping[b_dirx]

    ## find the missing direction
    done = np.zeros(3, dtype='bool')
    done[a_dir] = True
    done[b_dir] = True
    c_dir = np.where(~done)[0][0]

    a_lv = get_median(all_points_3d, bp_index[a_l])
    a_rv = get_median(all_points_3d, bp_index[a_r])
    b_lv = get_median(all_points_3d, bp_index[b_l])
    b_rv = get_median(all_points_3d, bp_index[b_r])

    a_diff = a_rv - a_lv
    b_diff = ortho(b_rv - b_lv, a_diff)

    M = np.zeros((3,3))
    M[a_dir] = a_diff
    M[b_dir] = b_diff
    # form a right handed coordinate system
    if (a_dir,b_dir) in [(0,1), (2,0), (1,2)]:
        M[c_dir] = np.cross(a_diff, b_diff)
    else:
        M[c_dir] = np.cross(b_diff, a_diff)

    M /= np.linalg.norm(M, axis=1)[:,None]

    center = get_median(all_points_3d, bp_index[ref_point])

    all_points_3d_adj = all_points_3d.dot(M.T)
    center_new = get_median(all_points_3d_adj, bp_index[ref_point])
    all_points_3d_adj = all_points_3d_adj - center_new

    return all_points_3d_adj, M, center_new
    
def proc(data_name,results_dir_root, config_path, n_kp, redo=False):
    result_dir = results_dir_root+'/' + data_name    
    if os.path.exists(os.path.dirname(config_path) + '/joint_len.npy'):
        if os.path.exists(result_dir + '/kp3d_fxdJointLen.pickle') & (not redo):
            print(f'Skip as exist:{data_name:s}/kp3d_fxdJointLen.pickle ')
            return    
    else :
        if os.path.exists(result_dir + '/kp3d.pickle') & (not redo):
            print(f'Skip as exist:{data_name:s}/kp3d.py')
            return    
    

    ##### make calibration files
    cfg = toml.load(open('./config_tmpl.toml'))
    cfg['model_folder'] = os.path.abspath(os.path.dirname(result_dir))
    toml.dump(cfg, open(result_dir + '/config.toml', mode='w'))

    calib = toml.load(open('./calibration_tmpl.toml'))
    file_intirin = os.path.dirname(config_path) + '/cam_intrinsic.h5'
    file_extirin = os.path.dirname(config_path) + '/cam_extrinsic_optim.h5'

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    for i, id in enumerate(ID):
        ID[i] = str(id)

    with h5py.File(file_intirin, 'r') as f:
        for i_cam, k in enumerate(ID):
            mtx = f[k]['mtx'][()]
            mtx[:2,:] /= 2
            dist = f[k]['dist'][()]
            xi=f[k]['xi'][()]
            K=f[k]['K'][()]
            D=f[k]['D'][()]
                    
            calib['cam_'+str(i_cam)]['matrix'] = mtx.tolist()
            calib['cam_'+str(i_cam)]['distortions'] = dist.ravel().tolist()
            calib['cam_'+str(i_cam)]['name'] = ID[i_cam]
            calib['cam_'+str(i_cam)]['xi'] = xi.ravel().tolist()
            calib['cam_'+str(i_cam)]['K'] = K.tolist()
            calib['cam_'+str(i_cam)]['D'] = D.ravel().tolist()  

    with h5py.File(file_extirin, 'r') as f:
        for i_cam, k in enumerate(ID):
            calib['cam_'+str(i_cam)]['rotation'] = f[k]['rvec'][()].ravel().tolist()
            calib['cam_'+str(i_cam)]['translation'] = f[k]['tvec'][()].ravel().tolist()    
            calib['cam_'+str(i_cam)]['name'] = ID[i_cam]     

    toml.dump(calib, open(result_dir + '/calibration.toml', mode='w'))

    ##### filter 2d key points
    print('##### 2D filtering....', flush=True)
    with open(result_dir + '/kp2d.pickle', 'rb') as f:
        kp2d = pickle.load(f)
    
    config=[]
    config = {"filter":{}}
    config['filter']['score_threshold']=0.3
    config['filter']['n_back']=3
    config['filter']['offset_threshold']=25
    config['filter']['multiprocessing']=True
    
    n_animal = kp2d.shape[0]
    n_frame = kp2d.shape[1]
    n_cam = kp2d.shape[2]

    kp2d = kp2d.transpose((1,3,0,4,2))

    kp2d_f = np.zeros(kp2d.shape, dtype=float) 
    
    for i_animal in range(n_animal):
        print('animal:', i_animal)
        for i_cam in range(n_cam):
            points = kp2d[:,:,i_animal,:,i_cam]
            points = np.expand_dims(points,2)
            points_f, scores_f = af.filter_pose_viterbi(config, points, [])
            points_f = af.wrap_points(points_f, scores_f)
            kp2d_f[:,:,i_animal,:,i_cam] = np.squeeze(points_f)

    with open(result_dir + '/kp2d_f.pickle', 'wb') as f:
        pickle.dump(kp2d_f , f)

    ##### reconstruct 3d keypoint
    
    print('##### 3D reconstruction....', flush=True)

    calib_fname = result_dir + '/calibration.toml'
    config_fname = result_dir + '/config.toml'

    if os.path.exists(os.path.dirname(config_path) + '/joint_len.npy'):
        joint_len = np.load(os.path.dirname(config_path) + '/joint_len.npy')
        joint_len_median = np.median(joint_len, axis=0)
    else:
        joint_len_median = None

    with open(result_dir + '/kp2d_f.pickle', 'rb') as f:
        kp2d_f = pickle.load(f)

    n_frame, n_kp, n_animal, _, n_cam = kp2d_f.shape

    kp2d_f = kp2d_f.transpose((2,4,0,1,3))

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    for i, id in enumerate(ID):
        ID[i] = str(id)

    config = toml.load(config_fname)

    # macaque model
    bodyparts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    # marmo model
    #bodyparts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
    #            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
    #            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
    #            'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'back']

    cgroup = CameraGroup.load(calib_fname)
    cgroup = cgroup.subset_cameras_names(ID)

    kp3d = np.zeros([n_animal, n_frame, n_kp, 3], dtype=float)
    E = np.zeros([n_animal, n_frame, n_kp], dtype=float)
    S = np.zeros([n_animal, n_frame, n_kp], dtype=float)
    joint_len = []
    for i_animal in range(n_animal):
        print('animal:', i_animal)

        all_points_raw = kp2d_f[i_animal,:,:,:,:2]
        all_scores = kp2d_f[i_animal,:,:,:,2]

        bad = all_scores < config['triangulation']['score_threshold']
        all_points_raw[bad] = np.nan

        if config['triangulation']['optim']:
            constraints = load_constraints(config, bodyparts)
            constraints_weak = load_constraints(config, bodyparts, 'constraints_weak')

            points_2d = all_points_raw
            scores_2d = all_scores

            points_shaped = points_2d.reshape(n_cam, n_frame*n_kp, 2)
            if config['triangulation']['ransac']:
                points_3d_init, _, _, _ = cgroup.triangulate_ransac(points_shaped, progress=True)
            else:
                points_3d_init = cgroup.triangulate(points_shaped, progress=True)
            points_3d_init = points_3d_init.reshape((n_frame, n_kp, 3))

            c = np.isfinite(points_3d_init[:, :, 0])
            if np.sum(c) < 20:
                print("warning: not enough 3D points to run optimization")
                points_3d = points_3d_init
            else:
                if joint_len_median is None:
                    points_3d, jl = cgroup.optim_points(
                        points_2d, points_3d_init,
                        constraints=constraints,
                        constraints_weak=constraints_weak,
                        #scores=scores_2d,
                        scale_smooth=config['triangulation']['scale_smooth'],
                        scale_length=config['triangulation']['scale_length'],
                        scale_length_weak=config['triangulation']['scale_length_weak'],
                        n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
                        reproj_error_threshold=config['triangulation']['reproj_error_threshold'],
                        verbose=True)
                else:
                    points_3d, jl = cgroup.optim_points_jointlenfix(
                        points_2d, points_3d_init, joint_len_median,
                        constraints=constraints,
                        constraints_weak=constraints_weak,
                        #scores=scores_2d,
                        scale_smooth=config['triangulation']['scale_smooth'],
                        scale_length=config['triangulation']['scale_length'],
                        scale_length_weak=config['triangulation']['scale_length_weak'],
                        n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
                        reproj_error_threshold=config['triangulation']['reproj_error_threshold'],
                        verbose=True)
                joint_len.append(jl)
                print(jl.shape)

            np.save(result_dir + '/joint_len.npy', np.array(joint_len))

            points_2d_flat = points_2d.reshape(n_cam, -1, 2)
            points_3d_flat = points_3d.reshape(-1, 3)

            errors = cgroup.reprojection_error(
                points_3d_flat, points_2d_flat, mean=True)
            good_points = ~np.isnan(all_points_raw[:, :, :, 0])
            num_cams = np.sum(good_points, axis=0).astype('float')

            all_points_3d = points_3d
            all_errors = errors.reshape(n_frame, n_kp)

            all_scores[~good_points] = 2
            scores_3d = np.min(all_scores, axis=0)

            scores_3d[num_cams < 1] = np.nan
            all_errors[num_cams < 1] = np.nan

        else:
            points_2d = all_points_raw.reshape(n_cam, n_frame*n_kp, 2)
            if config['triangulation']['ransac']:
                points_3d, picked, p2ds, errors = cgroup.triangulate_ransac(
                    points_2d, min_cams=3, progress=True)

                all_points_picked = p2ds.reshape(n_cam, n_frame, n_kp, 2)
                good_points = ~np.isnan(all_points_picked[:, :, :, 0])

                num_cams = np.sum(np.sum(picked, axis=0), axis=1)\
                                .reshape(n_frame, n_kp)\
                                .astype('float')
            else:
                points_3d = cgroup.triangulate(points_2d, progress=True)
                errors = cgroup.reprojection_error(points_3d, points_2d, mean=True)
                good_points = ~np.isnan(all_points_raw[:, :, :, 0])
                num_cams = np.sum(good_points, axis=0).astype('float')

            all_points_3d = points_3d.reshape(n_frame, n_kp, 3)
            all_errors = errors.reshape(n_frame, n_kp)

            all_scores[~good_points] = 2
            scores_3d = np.min(all_scores, axis=0)

            scores_3d[num_cams < 2] = np.nan
            all_errors[num_cams < 2] = np.nan
            num_cams[num_cams < 2] = np.nan

        if 'reference_point' in config['triangulation'] and 'axes' in config['triangulation']:
            all_points_3d_adj, M, center = correct_coordinate_frame(config, all_points_3d, bodyparts)
        else:
            all_points_3d_adj = all_points_3d
            M = np.identity(3)
            center = np.zeros(3)

        kp3d[i_animal,:,:,:] = all_points_3d_adj
        S[i_animal, :,:]  = scores_3d
        E[i_animal, :,:]  = all_errors

    data2 = {'kp3d':kp3d, 'kp3d_score':S, 'kp3d_err':E, 'joint_len':joint_len}

    if os.path.exists(os.path.dirname(config_path) + '/joint_len.npy'):    
        with open(result_dir + '/kp3d_fxdJointLen.pickle', 'wb') as f:
            pickle.dump(data2 , f)
    else: 
        with open(result_dir + '/kp3d.pickle', 'wb') as f:
            pickle.dump(data2 , f)

if __name__ == '__main__':
    
    pass