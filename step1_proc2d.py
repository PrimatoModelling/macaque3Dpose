import json
import time
import imgstore
import warnings
import glob
import os
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import platform

import torch

from mmcv import imcrop

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

from mmtrack.apis import inference_mot
from mmtrack.apis import init_model as init_tracking_model

from mmcls.apis import inference_model as inference_cls_model
from mmcls.apis import init_model as init_cls_model

def process_mmtracking_results(mmtracking_results):
    """Process mmtracking results.

    :param mmtracking_results:
    :return: a list of tracked bounding boxes
    """
    person_results = []
    # 'track_results' is changed to 'track_bboxes'
    # in https://github.com/open-mmlab/mmtracking/pull/300
    if 'track_bboxes' in mmtracking_results:
        tracking_results = mmtracking_results['track_bboxes'][0]
    elif 'track_results' in mmtracking_results:
        tracking_results = mmtracking_results['track_results'][0]

    for track in tracking_results:
        person = {}
        person['track_id'] = int(track[0])
        person['bbox'] = track[1:]
        person_results.append(person)
    return person_results

def get_nearest(a, A):
    d = np.abs(A-a)
    return np.argmin(d)

def proc(data_name, results_dir_root, raw_data_dir, device_str, fps, t_intv=None, redo=False, is_mff1y=False):
    
    ### models & parameters
    tracking_config = 'model/track/jm_bytetrack_yolov3.py'
    pose_config = 'model/pose/jm_hrnet_w32_256x256_2_b.py'
    pose_checkpoint = 'weight/pose.pth'

    if is_mff1y:
        id_config = 'model/id/jm_resnet50_8xb32_in1k_mff1y.py'
        id_checkpoint = 'weight/id_mff1y.pth'
    else: 
        id_config = 'model/id/jm_resnet50_8xb32_in1k.py'
        id_checkpoint = 'weight/id.pth'

    print(id_config, id_checkpoint)

    bbox_thr = 0.3
    vis_radius = 4
    vis_thickness = 2
    vis_kpt_thr = 0.3

    ### process videos

    L = glob.glob(raw_data_dir + '/' + data_name + '.*/metadata.yaml')
    L.sort()

    stores = []
    for l in L:
        store = imgstore.new_for_filename(l)
        stores.append(store)

    mdata = stores[0].get_frame_metadata()
    t = mdata['frame_time']

    if t_intv is None:
        with open(os.path.dirname(L[0]) + '/metadata.yaml') as f:
            fileinfo = yaml.safe_load(f)
        if 'trim_start' in fileinfo.keys():
            t_start = fileinfo['trim_start']
        else:
            t_start = t[0]
        t_end = t[-1]
    else:
        t_start = t[0]+t_intv[0]
        t_end = t[0]+t_intv[1]

    T = np.arange(t_start, t_end, 1/fps)

    def process_single_cam(i_store):    

        ## initialize models
        tracking_model = init_tracking_model(
            tracking_config, None, device=device_str.lower())

        pose_model = init_pose_model(
            pose_config, pose_checkpoint, device=device_str.lower())

        id_model = init_cls_model(id_config, id_checkpoint, device=device_str.lower())
        
        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            dataset_info = DatasetInfo(dataset_info)

        return_heatmap = False
        output_layer_names = None
                
        torch.set_num_threads(1)

        ## process frames

        store = stores[i_store]
        cname = os.path.basename(store.filename).split('.')[1]

        if t_intv is None:
            out_dir = results_dir_root+'/' + data_name + '/' + cname
        else:
            out_dir = results_dir_root+'/' + data_name + '.{:04d}-{:04d}'.format(int(t_intv[0]), int(t_intv[1])) + '/' + cname
        path_json_alldata = out_dir + '/alldata.json'
        path_npy_frame_num = out_dir + '/frame_num.npy'
        if os.path.exists(path_json_alldata) & os.path.exists(path_npy_frame_num) & (not redo):
            return 

        mdata = store.get_frame_metadata()
        t_cam = mdata['frame_time']
        frame_num = mdata['frame_number']

        result = []
        F = []
        
        pre_result = None
        frame_cnt = 0
        frame_number = -1

        for t in tqdm(T):

            i = get_nearest(t, t_cam)

            if frame_number >= frame_num[i]:
                result.append(pre_result)
                F.append(frame_number)
                continue

            if frame_number == -1:
                img, (frame_number, frame_time) = store.get_image(frame_num[i])
            else:
                while frame_number < frame_num[i]:
                    img, (frame_number, frame_time) = store.get_next_image()

            # detection & tracking
            s = time.time()
            mmtracking_results = inference_mot(
                tracking_model, img, frame_id=frame_cnt)
            person_results = process_mmtracking_results(mmtracking_results)
            #print('detection+tracking (ms):', (time.time()-s)*1000)

            # keypoint estimation
            s = time.time()
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)
            #print('pose (ms):', (time.time()-s)*1000)

            # check id
            s = time.time()
            #print(len(pose_results))
            #if len(pose_results)==0:
            #    print('no detection!')
            if len(pose_results) > 0:
                bboxes = []
                for p in pose_results:
                    bboxes.append(p['bbox'][:4])
                bboxes = np.array(bboxes)
                patches = imcrop(img, bboxes)
                for i_pose, patch in enumerate(patches):
                    id_result = inference_cls_model(id_model, patch)
                    pose_results[i_pose]['id'] = id_result

            #print('id (ms):', (time.time()-s)*1000)

            result.append(pose_results)
            F.append(frame_number)

            pre_result = pose_results
            frame_cnt += 1

        F = np.array(F, dtype=int)
 
        os.makedirs(out_dir, exist_ok=True)

        np.save(path_npy_frame_num, F)

        data = []
        for i_frame, t in enumerate(result):
            data.append([])
            for i_box, tt in enumerate(t):
                d = [tt['track_id'], tt['bbox'][0], tt['bbox'][1], tt['bbox'][2], tt['bbox'][3], 
                    tt['keypoints'].tolist(), int(tt['id']['pred_label']), tt['id']['pred_score']]
                data[i_frame].append(d)
        
        with open(path_json_alldata, 'w') as f:
            json.dump(data, f)

    for i_store in range(len(stores)):
        process_single_cam(i_store)

if __name__ == '__main__':

    pass
