import numpy as np
import cv2
import glob
import os
import h5py
import yaml
import os
from cv2 import aruco
from tqdm import tqdm
import scipy.io
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import math
import imgstore
import json
import shutil
import random

def analyze_chessboardvid(config_path, saveimg=False, frame_intv=5):
    if saveimg:
        os.makedirs('./tmp/', exist_ok=True)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    square_size = cfg['chessboard_square_size']

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*square_size

    vid_dir = os.path.dirname(config_path) + '/' + cfg['chessboard_vid_folder']
    ID = cfg['camera_id']

    with h5py.File(os.path.dirname(config_path) + '/chessboard_points.h5', mode='w') as h5file:
        for id in ID:
            vf = glob.glob(vid_dir + '/' + str(id) + '*.mp4')[0]

            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.
            cap = cv2.VideoCapture(vf)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(id)
            for i_frame in tqdm(range(10,frame_count,frame_intv)):

                cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame)

                ret, frame = cap.read()

                # Find the chess board corners
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (9,6))
                if ret == True:
                    
                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    imgpoints.append(corners2)
                    objpoints.append(objp)

                    if saveimg:
                        frame2 = cv2.drawChessboardCorners(frame, (9,6), corners2,ret)
                        outputfname = "./tmp/" + str(id) + '_' + str(i_frame) + ".jpg"
                        frame3 = cv2.resize(frame2, (640, 480))
                        cv2.imwrite(outputfname, frame3)

            h5file.create_dataset('/'+str(id)+'/imp', data = imgpoints)
            h5file.create_dataset('/'+str(id)+'/objp', data = objpoints)

def calibrate_intrinsic(config_path, mtx_init=None, dist_init=None):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']
    imsize = cfg['img_size']

    with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='w') as h5file_out:
        for id in ID:

            with h5py.File(os.path.dirname(config_path) + '/chessboard_points.h5', mode='r') as h5file:
                imgpoints = h5file['/'+str(id)+'/imp'][()]
                objpoints = h5file['/'+str(id)+'/objp'][()]

            # normal camera
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (imsize[0], imsize[1]) , mtx_init, dist_init)

            h5file_out.create_dataset('/'+str(id)+'/mtx', data = mtx)
            h5file_out.create_dataset('/'+str(id)+'/dist', data = dist)

            # omnidir camera
            imgpoints2 = []
            objpoints2 = []
            for i in range(imgpoints.shape[0]):
                imgpoints2.append(imgpoints[i,:,:,:])
                objpoints2.append(np.reshape(objpoints[i,:,:], [-1,1,3]))
            
            calibration_flags = cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER

            rms, K, xi, D, rvecs, tvecs, idx = \
                                cv2.omnidir.calibrate(
                                    objpoints2,
                                    imgpoints2,
                                    (imsize[0], imsize[1]),
                                    K=None,
                                    xi=None,
                                    D=None,
                                    flags=calibration_flags,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
                                )

            h5file_out.create_dataset('/'+str(id)+'/K', data = K)
            h5file_out.create_dataset('/'+str(id)+'/xi', data = xi)
            h5file_out.create_dataset('/'+str(id)+'/D', data = D)

def label_cagekeypoints(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    kp_file = os.path.dirname(config_path) + '/' + cfg['cagekeypoint_position']
    ID = cfg['camera_id']

    kps = np.loadtxt(kp_file,  delimiter=',')

    wname = 'label cage keypoints'
    crnt_kp = 0
    n_kp = kps.shape[0]
    points = np.zeros([n_kp, 3], dtype=np.int)
    img = []

    data = {}
    path_annotation = os.path.dirname(config_path) + '/cagepoints_annotation.h5'
    if os.path.exists(path_annotation):
        with h5py.File(path_annotation, mode='r') as f_cagepoints:
            for k in f_cagepoints.keys():
                d = f_cagepoints['/'+k][()]
                data[k] = d
    else:
        for id in ID:
            data[str(id)] = np.zeros([n_kp, 6])

    def add_point(x, y):
        points[crnt_kp, :] = np.array([1, x, y])

    def remove_point():
        points[crnt_kp, 0] = 0

    def update_disp():
        img2 = np.copy(img)
        cv2.putText(img2, 'kp: ' + str(crnt_kp), (0, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3, cv2.LINE_AA)
        for i_p in range(n_kp):
            if points[i_p, 0] > 0:
                cv2.putText(img2, str(i_p), (points[i_p, 1], points[i_p, 2]+20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(img2, (points[i_p, 1], points[i_p, 2]), (0,0,255), thickness=2, markerSize=15)
        cv2.imshow(wname, img2)

    def onMouse(event, x, y, flag, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            add_point(x, y)
            update_disp()
        if event == cv2.EVENT_MBUTTONDOWN:
            remove_point()
            update_disp()

    vid_dir = os.path.dirname(config_path) + '/' + cfg['cagekeypoint_vid_folder']

    for id in ID:
        points = data[str(id)][:,:3].astype(np.int)

        vf = glob.glob(vid_dir + '/*' + str(id) + '*.mp4')[0]

        cap = cv2.VideoCapture(vf)
        ret, frame = cap.read()
        img = cv2.resize(frame, (640,480))

        cv2.namedWindow(wname)
        cv2.setMouseCallback(wname, onMouse, [])
        update_disp()
        while True:
            k = cv2.waitKey()
            if k == 97: #A
                prev = max(cap.get(cv2.CAP_PROP_POS_FRAMES) - 10, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, prev)
                ret, frame = cap.read()
                img = cv2.resize(frame, (640,480))
                update_disp()
            if k == 100: #D
                ret, frame = cap.read()
                img = cv2.resize(frame, (640,480))
                update_disp()
            if k == 119: #W
                crnt_kp = min(n_kp-1, crnt_kp+1)
                update_disp()
            if k == 115: #S
                crnt_kp = max(0, crnt_kp-1)
                update_disp()

            if k == 32: #space key
                break

        d = np.hstack((points, kps))
        data[str(id)] = d
        
    with h5py.File(path_annotation, mode='w') as f_cagepoints:
        for id in ID:
            f_cagepoints.create_dataset('/'+str(id), data = data[str(id)])


    cv2.destroyAllWindows()

def get_extrinsic_from_cagekeypoints(config_path):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    ID = cfg['camera_id']

    with h5py.File(os.path.dirname(config_path) + '/cagepoints_annotation.h5', mode='r') as f_cagedim:
        with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='r') as f_intrin:
            with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic.h5', mode='w') as f_extrin:
                for id in ID:
                
                    mtx = f_intrin['/'+str(id)+'/mtx'][()]
                    dist = f_intrin['/'+str(id)+'/dist'][()]
                    cp = f_cagedim['/'+str(id)][()]

                    cp = cp[cp[:,0]>0,1:]

                    imgp = cp[:,0:2]*2048/640-0.0
                    objp = cp[:,2:]-0.0

                    ret, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

                    f_extrin.create_dataset('/'+str(id)+'/rvec', data = rvecs)
                    f_extrin.create_dataset('/'+str(id)+'/tvec', data = tvecs)

                    rotM = cv2.Rodrigues(rvecs)[0]
                    camerapos = -np.matrix(rotM).T * np.matrix(tvecs)
                    print('3D pos of camera ' + str(id))
                    print(camerapos)

def analyze_aruco_marker_vid(config_path, disp_result=False):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    marker_len = cfg['marker_size']
    vid_dir = os.path.dirname(config_path) + '/' + cfg['marker_vid_folder']

    with h5py.File(os.path.dirname(config_path) + '/marker_trace.h5', mode='w') as f_aruco:
       
        ID = cfg['camera_id']

        for id in ID:
            
            print(id)

            vf = glob.glob(vid_dir + '/*' + str(id) + '*/*.mp4')[0]

            with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='r') as f_ci:
                mtx = f_ci['/'+str(id)+'/mtx'][()]
                dist = f_ci['/'+str(id)+'/dist'][()]

            dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

            cap = cv2.VideoCapture(vf)

            w = 640
            h = 480
            rsize_ratio = 0

            C = []
            while True:
                ret, frame = cap.read()

                if frame is None: 
                    break

                rsize_ratio = frame.shape[1]/w
                frame_r = cv2.resize(frame, (w,h))

                corners, mrk_ids, rejectedImgPoints = aruco.detectMarkers(frame_r, dict_aruco) 
                aruco.drawDetectedMarkers(frame_r, corners, mrk_ids, (0,255,255)) 

                if mrk_ids is not None:

                    corner = corners[0]
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corner*rsize_ratio, marker_len, mtx, dist)
                    imgp, jac = cv2.projectPoints(np.zeros([1,3]), rvecs, tvecs, mtx, dist)
                    imgp = np.squeeze(imgp)
                    cv2.drawMarker(frame_r, (int(imgp[0]/rsize_ratio), int(imgp[1]/rsize_ratio)), (255,0,0), thickness=3)
                    C.append([imgp[0], imgp[1]])
                else:
                    C.append([-1, -1])

                if disp_result:
                    cv2.imshow('preview', frame_r)
                    cv2.waitKey(1)

            C = np.array(C, dtype=np.float)

            f_aruco.create_dataset('/'+str(id), data = C)

    cv2.destroyAllWindows()

def analyze_aruco_cube_vid(config_path, frame_intv=5, disp_result=False, fps=24):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    marker_len = cfg['marker_size']
    vid_dir = os.path.dirname(config_path) + '/' + cfg['marker_vid_folder']
    cube_len = cfg['cube_size']
    arucomrk_to_cubecenter = np.array([[0.0, 0.0, -cube_len/2]])

    with h5py.File(os.path.dirname(config_path) + '/marker_trace.h5', mode='w') as f_aruco:
       
        ID = cfg['camera_id']

        S = []
        for id in ID:
            mdata = glob.glob(vid_dir + '/*' + str(id) + '*/metadata.yaml')[0]
            store = imgstore.new_for_filename(mdata)
            frame, (frame_number, frame_timestamp) = store.get_next_image()
            S.append(store)
        
        t0 = frame_timestamp
        duration = (S[0].frame_max - S[0].frame_min) / fps
        frame_range = range(fps*5, int(duration*fps) - fps*5, frame_intv)

        for i_cam, id in enumerate(ID):

            print(id)

            with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='r') as f_ci:
                mtx = f_ci['/'+str(id)+'/mtx'][()]
                dist = f_ci['/'+str(id)+'/dist'][()]

            dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

            w = 640
            h = 480
            rsize_ratio = 0

            C = []
            for i_frame in frame_range:
                
                frame, (frame_number, frame_timestamp) = S[i_cam].get_nearest_image(t0+i_frame/fps)

                if frame is None: 
                    break

                rsize_ratio = frame.shape[1]/w
                frame_r = cv2.resize(frame, (w,h))

                corners, mrk_ids, rejectedImgPoints = aruco.detectMarkers(frame_r, dict_aruco) 
                aruco.drawDetectedMarkers(frame_r, corners, mrk_ids, (0,255,255)) 

                if mrk_ids is not None:
                    P = []
                    for i, corner in enumerate(corners):
                        #if 23506237 == id or 23506239 == id or 23511613 == id or 23511614 == id:
                        #    if mrk_ids[i][0] == 0:
                        #        continue
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corner*rsize_ratio, marker_len, mtx, dist)
                        imgp, jac = cv2.projectPoints(arucomrk_to_cubecenter, rvecs, tvecs, mtx, dist)
                        imgp = np.squeeze(imgp)
                        marker_c = np.mean(corner[0], axis=0)
                        if np.linalg.norm(imgp/rsize_ratio-marker_c) < w/32:#w/5:
                            P.append(imgp)
                        else:
                            print('warning: cube center estimation ignored; too far from square center')

                    if len(P) > 0:
                        imgp_m = np.mean(np.vstack(P), axis=0)
                        cv2.drawMarker(frame_r, (int(imgp_m[0]/rsize_ratio), int(imgp_m[1]/rsize_ratio)), (0,0,255), thickness=2)
                        C.append([imgp[0], imgp[1]])
                    else:
                        C.append([-1, -1])
                else:
                    C.append([-1, -1])

                if disp_result:
                    cv2.imshow('preview', frame_r)
                    cv2.waitKey(1)

            C = np.array(C, dtype=np.float)

            f_aruco.create_dataset('/'+str(id), data = C)

    cv2.destroyAllWindows()

def undistortPoints(config_path, pos_2d, omnidir=False, camparam=None):
    
    if camparam is None:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        ID = cfg['camera_id']
    else:
        ID = camparam['camera_id']

    pos_2d_undist = []

    if omnidir:
        for i_cam, id in enumerate(ID):
            
            if camparam is None:
                with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='r') as f_intrin:
                    K = f_intrin['/'+str(id)+'/K'][()]
                    xi = f_intrin['/'+str(id)+'/xi'][()]
                    D = f_intrin['/'+str(id)+'/D'][()]
            else:
                K = camparam['K'][i_cam]
                xi = camparam['xi'][i_cam]
                D = camparam['D'][i_cam]

            p = pos_2d[i_cam]+0.0
            p_undist = cv2.omnidir.undistortPoints(np.array([p.tolist()], np.float), K, D, xi, np.eye(3))
            p_undist = np.squeeze(p_undist)
            pos_2d_undist.append(p_undist)
    else:
        for i_cam, id in enumerate(ID):
            with h5py.File(os.path.dirname(config_path) + '/cam_intrinsic.h5', mode='r') as f_intrin:
                mtx = f_intrin['/'+str(id)+'/mtx'][()]
                dist = f_intrin['/'+str(id)+'/dist'][()]
            p = pos_2d[i_cam]+0.0
            p_undist = cv2.undistortPoints(p, mtx, dist)
            p_undist = np.squeeze(p_undist)
            pos_2d_undist.append(p_undist)

    return pos_2d_undist

def triangulatePoints(config_path, pos_2d_undist, frame_use, use_optim_extrin, camparam=None):

    if camparam is None:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        ID = cfg['camera_id']

        if use_optim_extrin:
            path_extrin = os.path.dirname(config_path) + '/cam_extrinsic_optim.h5'
        else:
            path_extrin = os.path.dirname(config_path) + '/cam_extrinsic.h5'

        pmat = []
        for i_cam, id in enumerate(ID):
            with h5py.File(path_extrin, mode='r') as f_extrin:
                rvecs = f_extrin['/'+str(id)+'/rvec'][()]
                tvecs = f_extrin['/'+str(id)+'/tvec'][()]
                rmtx, jcb = cv2.Rodrigues(rvecs)
                R = np.hstack([rmtx, tvecs])
                pmat.append(R)
    
    else:
        ID = camparam['camera_id']
        pmat = camparam['pmat']

    n_frame = frame_use.shape[0]
    n_cam = frame_use.shape[1]

    P = np.zeros((n_frame,3))
    U = pos_2d_undist

    for i_frame in range(n_frame):
        if np.sum(frame_use[i_frame, :]) < 2:
            P[i_frame, :] = np.nan
            continue

        A = []
        for i_cam in range(n_cam):
            if frame_use[i_frame, i_cam]:
                a1 = U[i_cam][i_frame,0]*pmat[i_cam][2,:] - pmat[i_cam][0,:]
                a2 = U[i_cam][i_frame,1]*pmat[i_cam][2,:] - pmat[i_cam][1,:]
                a = np.vstack((a1, a2))
                A.append(a)
        
        A = np.vstack(A)

        b = A[:,3]
        a = A[:, :3]
        a_inv = np.linalg.pinv(a)
        X = np.matmul(a_inv, b)

        P[i_frame, :] = -X

    return P

def optimize_extrinsic(config_path, show_estimated_campos=True, omnidir=False, fixcam0=True):
    # optimize extrinsic parameters using bundle adjustment
    # see: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    pos_2d = []
    with h5py.File(os.path.dirname(config_path) + '/marker_trace.h5', mode='r') as f_mrkt:
        for id in ID:
            pos_2d.append(f_mrkt['/' + str(id)][()])

    n_frame = pos_2d[0].shape[0]-5
    n_cam = len(ID)
    frame_use = np.zeros((n_frame, n_cam), dtype=np.bool)
    for i_frame in range(n_frame):
        for i_cam in range(n_cam):
            if pos_2d[i_cam][i_frame, 0] >= 0.0:
                frame_use[i_frame, i_cam] = True

    pos_2d_undist = undistortPoints(config_path, pos_2d, omnidir)

    P = triangulatePoints(config_path, pos_2d_undist, frame_use, False)

    #### prepare parameters for bundle adjustment
    camera_params = np.zeros((n_cam,6))

    for i_cam, id in enumerate(ID):
        with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic.h5', mode='r') as f_extrin:
            rvecs = f_extrin['/'+str(id)+'/rvec'][()]
            tvecs = f_extrin['/'+str(id)+'/tvec'][()]
        camera_params[i_cam, :3] = rvecs[:,0]
        camera_params[i_cam, 3:6] = tvecs[:,0]

    I_frame_use = np.argwhere(np.sum(frame_use, axis=1) >= 2).ravel()
    p_3d = P[I_frame_use, :]
    p_2d = []
    for i_cam in range(n_cam):
        p_2d.append(pos_2d_undist[i_cam][I_frame_use, :])
    frame_use2 = frame_use[I_frame_use, :]

    points_2d = []
    point_indices = []
    camera_indices = []
    for i_cam in range(n_cam):
        for i_frame in range(p_3d.shape[0]):
            if frame_use2[i_frame, i_cam]:
                camera_indices.append(i_cam)
                point_indices.append(i_frame)
                points_2d.append(p_2d[i_cam][i_frame,:])

    points_2d = np.vstack(points_2d)
    points_3d = p_3d
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 6 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    def rotate(points, rot_vecs):
        """Rotate points by given rotation vectors.
        
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    def project(points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        #f = camera_params[:, 6]
        #k1 = camera_params[:, 7]
        #k2 = camera_params[:, 8]
        #n = np.sum(points_proj**2, axis=1)
        #r = 1 + k1 * n + k2 * n**2
        #points_proj *= (r * f)[:, np.newaxis]
        return points_proj

    def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.
        
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params2 = params[:n_cameras * 6].reshape((n_cameras, 6))
        if fixcam0: 
            camera_params2[0,:] = camera_params[0,:]
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = project(points_3d[point_indices], camera_params2[camera_indices])
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-5, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    
    cp = res.x[:camera_params.size].reshape((-1,6))
    with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic_optim.h5', mode='w') as f_extrin_opt:
        for i_cam, id in enumerate(ID):
            rvecs = cp[i_cam, :3].ravel()
            tvecs = cp[i_cam, 3:6].ravel()
            tvecs = tvecs[:, np.newaxis]
            f_extrin_opt.create_dataset('/'+str(id)+'/rvec', data = rvecs)
            f_extrin_opt.create_dataset('/'+str(id)+'/tvec', data = tvecs)

    if show_estimated_campos:
        with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic_optim.h5', mode='r') as f_extrin_opt:
            for i_cam, id in enumerate(ID):
                rvecs = f_extrin_opt['/'+str(id)+'/rvec'][()]
                tvecs = f_extrin_opt['/'+str(id)+'/tvec'][()]

                rotM = cv2.Rodrigues(rvecs)[0]
                camerapos = -np.matrix(rotM).T * np.matrix(tvecs)
                print(id, ':', np.array(camerapos).ravel())

    # For debugging
    P1 = triangulatePoints(config_path, pos_2d_undist, frame_use, False)
    P2 = triangulatePoints(config_path, pos_2d_undist, frame_use, True)
    scipy.io.savemat('test.mat', {'P1':P1, 'P2':P2})

def optimize_all_camera_params(config_path, show_estimated_campos=True, omnidir=False, 
                                fixcam0=True, n_random_sample=-1, ftol=1e-3, max_nfev=None,
                                mode_eval=False, file_intrin=None, file_extrin=None, verbose=2):
    # optimize extrinsic parameters using bundle adjustment
    # see: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    pos_2d = []
    with h5py.File(os.path.dirname(config_path) + '/marker_trace.h5', mode='r') as f_mrkt:
        for id in ID:
            pos_2d.append(f_mrkt['/' + str(id)][()])

    n_frame = pos_2d[0].shape[0]-5
    n_cam = len(ID)
    frame_use = np.zeros((n_frame, n_cam), dtype=np.bool)
    for i_frame in range(n_frame):
        for i_cam in range(n_cam):
            if pos_2d[i_cam][i_frame, 0] >= 0.0:
                frame_use[i_frame, i_cam] = True

    
    if n_random_sample > 0:
        I = random.sample(list(range(n_frame)), n_random_sample)
        for i_cam in range(n_cam):
            pos_2d[i_cam] = pos_2d[i_cam][I,:]
        frame_use = frame_use[I,:]

    pos_2d_undist = undistortPoints(config_path, pos_2d, omnidir)

    P = triangulatePoints(config_path, pos_2d_undist, frame_use, False)

    #### prepare parameters for bundle adjustment
    camera_params = np.zeros((n_cam,16))

    if file_extrin is None:
        file_extrin = os.path.dirname(config_path) + '/cam_extrinsic.h5'
    if file_intrin is None:
        file_intrin = os.path.dirname(config_path) + '/cam_intrinsic.h5'

    for i_cam, id in enumerate(ID):
        with h5py.File(file_extrin, mode='r') as f_extrin:
            rvecs = f_extrin['/'+str(id)+'/rvec'][()]
            tvecs = f_extrin['/'+str(id)+'/tvec'][()]
        if rvecs.ndim == 1:
            rvecs = rvecs[:, np.newaxis]
        camera_params[i_cam, :3] = rvecs[:,0]
        camera_params[i_cam, 3:6] = tvecs[:,0]
        with h5py.File(file_intrin, mode='r') as f_intrin:
            K = f_intrin['/'+str(id)+'/K'][()]
            xi = f_intrin['/'+str(id)+'/xi'][()]
            D = f_intrin['/'+str(id)+'/D'][()]
        camera_params[i_cam, 6:11] = np.array([K[0,0], K[0,1], K[0,2] ,K[1,1], K[1,2]])
        camera_params[i_cam, 11] = xi[0,0]
        camera_params[i_cam, 12:16] = D.ravel()

    I_frame_use = np.argwhere(np.sum(frame_use, axis=1) >= 2).ravel()
    p_3d = P[I_frame_use, :]
    p_2d = []
    for i_cam in range(n_cam):
        p_2d.append(pos_2d[i_cam][I_frame_use, :])
    frame_use2 = frame_use[I_frame_use, :]

    points_2d = []
    point_indices = []
    camera_indices = []
    for i_cam in range(n_cam):
        for i_frame in range(p_3d.shape[0]):
            if frame_use2[i_frame, i_cam]:
                camera_indices.append(i_cam)
                point_indices.append(i_frame)
                points_2d.append(p_2d[i_cam][i_frame,:])

    points_2d = np.vstack(points_2d)
    points_3d = p_3d
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 16 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    def project(points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = np.zeros([points.shape[0],2])
        for i_point in range(points.shape[0]):
            rvec = camera_params[i_point, :3]
            tvec = camera_params[i_point, 3:6]
            k = camera_params[i_point, 6:11]
            K = np.array([[k[0], k[1], k[2]], [0, k[3], k[4]], [0, 0, 1]])
            xi = camera_params[i_point, 11]
            D = camera_params[i_point, 12:16]
            D = D[np.newaxis, :]
            p, jac = cv2.omnidir.projectPoints(np.reshape(points[i_point], (-1,1,3)), rvec, tvec, K, xi, D)
            points_proj[i_point, :] = p

        return points_proj

    def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.
        
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params2 = params[:n_cameras * 16].reshape((n_cameras, 16))
        if fixcam0: 
            camera_params2[0,0:6] = camera_params[0,0:6]
        points_3d = params[n_cameras * 16:].reshape((n_points, 3))
        points_proj = project(points_3d[point_indices], camera_params2[camera_indices])
        
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 16 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(16):
            A[2 * i, camera_indices * 16 + s] = 1
            A[2 * i + 1, camera_indices * 16 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 16 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 16 + point_indices * 3 + s] = 1

        return A

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    if mode_eval:
        return f0

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    res = least_squares(fun, x0, jac_sparsity=A, verbose=verbose, x_scale='jac', ftol=ftol, method='trf', loss='linear', max_nfev=max_nfev,
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    
    cp = res.x[:camera_params.size].reshape((-1,16))
    with h5py.File(os.path.dirname(file_extrin) + '/cam_extrinsic_optim.h5', mode='w') as f_extrin_opt:
        for i_cam, id in enumerate(ID):
            rvecs = cp[i_cam, :3].ravel()
            tvecs = cp[i_cam, 3:6].ravel()
            tvecs = tvecs[:, np.newaxis]
            f_extrin_opt.create_dataset('/'+str(id)+'/rvec', data = rvecs)
            f_extrin_opt.create_dataset('/'+str(id)+'/tvec', data = tvecs)

    with h5py.File(os.path.dirname(file_intrin) + '/cam_intrinsic.h5', mode='r') as f_intrin:
        with h5py.File(os.path.dirname(file_intrin) + '/cam_intrinsic_optim.h5', mode='w') as f_intrin_opt:
            for i_cam, id in enumerate(ID):
                k = cp[i_cam, 6:11].ravel()
                K = np.array([[k[0], k[1], k[2]], [0, k[3], k[4]], [0, 0, 1]])
                xi = np.array([[cp[i_cam, 11]]])
                D = cp[i_cam, 12:16]
                D = D[np.newaxis, :]

                f_intrin_opt.create_dataset('/'+str(id)+'/K', data = K)
                f_intrin_opt.create_dataset('/'+str(id)+'/xi', data = xi)
                f_intrin_opt.create_dataset('/'+str(id)+'/D', data = D)
                mtx = f_intrin['/'+str(id)+'/mtx'][()]
                dist = f_intrin['/'+str(id)+'/dist'][()]
                f_intrin_opt.create_dataset('/'+str(id)+'/mtx', data = mtx)
                f_intrin_opt.create_dataset('/'+str(id)+'/dist', data = dist)
                if show_estimated_campos:
                    print(id)
                    print(K)
                    print(xi)
                    print(D)


    if show_estimated_campos:
        with h5py.File(os.path.dirname(file_extrin) + '/cam_extrinsic_optim.h5', mode='r') as f_extrin_opt:
            for i_cam, id in enumerate(ID):
                rvecs = f_extrin_opt['/'+str(id)+'/rvec'][()]
                tvecs = f_extrin_opt['/'+str(id)+'/tvec'][()]

                rotM = cv2.Rodrigues(rvecs)[0]
                camerapos = -np.matrix(rotM).T * np.matrix(tvecs)
                print(id, ':', np.array(camerapos).ravel())

    # For debugging
    P1 = triangulatePoints(config_path, pos_2d_undist, frame_use, False)
    P2 = triangulatePoints(config_path, pos_2d_undist, frame_use, True)
    scipy.io.savemat('test.mat', {'P1':P1, 'P2':P2})

def extract_frames_for_3dannotation(config_path, video_path, out_dir, n_frame_extract=10, n_animal=1, n_kp=20, fps=24, mdl=None, frame_ts=None):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    S = []
    for id in ID:
        mdata = glob.glob(video_path + '.' + str(id) + '*/metadata.yaml')[0]
        store = imgstore.new_for_filename(mdata)
        frame, (frame_number, frame_timestamp) = store.get_next_image()
        S.append(store)
    
    t0 = frame_timestamp

    n_cam = len(ID)

    if frame_ts is not None:    # extract frames at specified time stamps
        os.makedirs(out_dir, exist_ok=True)
        for t in tqdm(frame_ts):
            TS = []
            for i_cam, store in enumerate(S):
                frame, (frame_number, frame_timestamp) = store.get_nearest_image(t)
                TS.append(frame_timestamp)
                frame2 = np.array(frame)
                cv2.imwrite(out_dir+'/{:08}.{}.jpg'.format(int((t-t0)*1000), ID[i_cam]), frame2)
            TS = np.array(TS)
            if np.max(TS)-np.min(TS) > 0.001:
                print('warning: sync is not good, time:', t-t0)

            kp_2d = np.zeros([n_animal, n_cam,n_kp,2])
            kp_2d[:,:,:,:] = np.nan

            kp_3d = np.zeros([n_animal, n_kp,3])
            kp_3d[:,:,:] = np.nan

            d = {'keypoints_2d': kp_2d.tolist(), 'keypoints_3d': kp_3d.tolist()}
            with open(out_dir+'/{:08}.json'.format(int((t-t0)*1000), ID[i_cam]), 'w') as fp:
                json.dump(d, fp)
    else:
        n_frame = np.zeros(n_cam)
        for i_cam, store in enumerate(S):
            n_frame[i_cam] = store.frame_count
        
        frame_start = 100
        n_frame = int(np.min(n_frame))
        d = (n_frame-frame_start) / n_frame_extract
        frames = np.arange(frame_start, n_frame, d)
        frames = frames.astype(np.int).tolist()
        
        os.makedirs(out_dir, exist_ok=True)
        for i_frame in tqdm(frames):
            t = i_frame/fps
            TS = []
            for i_cam, store in enumerate(S):
                frame, (frame_number, frame_timestamp) = store.get_nearest_image(t0+t)
                TS.append(frame_timestamp)
                frame2 = np.array(frame)
                cv2.imwrite(out_dir+'/{:08}.{}.jpg'.format(i_frame, ID[i_cam]), frame2)
            TS = np.array(TS)
            if np.max(TS)-np.min(TS) > 0.001:
                print('warning: sync is not good, frame no:', i_frame)

            kp_2d = np.zeros([n_animal, n_cam,n_kp,2])
            kp_2d[:,:,:,:] = np.nan

            kp_3d = np.zeros([n_animal, n_kp,3])
            kp_3d[:,:,:] = np.nan

            d = {'keypoints_2d': kp_2d.tolist(), 'keypoints_3d': kp_3d.tolist()}
            with open(out_dir+'/{:08}.json'.format(i_frame), 'w') as fp:
                json.dump(d, fp)

    names = []
    models = []
    for i in range(n_animal):
        names.append(f'individual{i+1}')
        models.append(mdl)

    noalias_dumper = yaml.dumper.Dumper
    noalias_dumper.ignore_aliases = lambda self, data: True
    with open(out_dir+'/metadata.yaml', 'w') as fp:
        d = {'n_animal': n_animal, 'n_cam': n_cam, 'n_kp': n_kp, 'animal_names': names, 'model':models }
        yaml.dump(d, fp, Dumper=noalias_dumper)

    os.makedirs(out_dir + '/calib', exist_ok=True)
    if os.path.exists(os.path.dirname(config_path) + '/cam_intrinsic_optim.h5'):
        shutil.copyfile(os.path.dirname(config_path) + '/cam_intrinsic_optim.h5', out_dir + '/calib/cam_intrinsic.h5')
    else:
        shutil.copyfile(os.path.dirname(config_path) + '/cam_intrinsic.h5', out_dir + '/calib/cam_intrinsic.h5')
        
    shutil.copyfile(os.path.dirname(config_path) + '/cam_extrinsic_optim.h5', out_dir + '/calib/cam_extrinsic_optim.h5')
    shutil.copyfile(os.path.dirname(config_path) + '/cam_extrinsic.h5', out_dir + '/calib/cam_extrinsic.h5')
    shutil.copyfile(os.path.dirname(config_path) + '/config.yaml', out_dir + '/calib/config.yaml')


"""
def applytransform(rvec1, tvec1, rvec2, tvec2, inv=False):

    rotM1 = cv2.Rodrigues(rvec1)[0]
    M1 = np.hstack([rotM1, tvec1])
    M1 = np.vstack([M1, np.array([0,0,0,1])])
    rotM2 = cv2.Rodrigues(rvec2)[0]
    M2 = np.hstack([rotM2, tvec2])
    M2 = np.vstack([M2, np.array([0,0,0,1])])

    if inv == True:
        M1 = np.linalg.pinv(M1)

    M = np.matmul( M1, M2)

    rvec = cv2.Rodrigues(M[:3,:3])[0]
    tvec = M[:3,3]
    tvec = tvec[:,np.newaxis]

    return rvec, tvec

def fix_extrinsic_optim(config_path, ref=0):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    ID = cfg['camera_id']

    id = ID[ref]
    with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic.h5', mode='r') as f_extrin_opt:
        rvecs_r1 = f_extrin_opt['/'+str(id)+'/rvec'][()]
        tvecs_r1 = f_extrin_opt['/'+str(id)+'/tvec'][()]
    with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic_optim.h5', mode='r') as f_extrin_opt:
        rvecs_r2 = f_extrin_opt['/'+str(id)+'/rvec'][()]
        tvecs_r2 = f_extrin_opt['/'+str(id)+'/tvec'][()]

    for id in ID:
        with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic_optim.h5', mode='r') as f_extrin_opt:
            rvecs = f_extrin_opt['/'+str(id)+'/rvec'][()]
            tvecs = f_extrin_opt['/'+str(id)+'/tvec'][()]

        rotM = cv2.Rodrigues(rvecs)[0]
        camerapos = -np.matrix(rotM).T * np.matrix(tvecs)
        print(id, ' (before):', np.array(camerapos).ravel())

        rvecs, tvecs = applytransform(rvecs_r2, tvecs_r2, rvecs, tvecs, inv=True)
        rvecs, tvecs = applytransform(rvecs_r1, tvecs_r1, rvecs, tvecs, inv=False)

        with h5py.File(os.path.dirname(config_path) + '/cam_extrinsic_optim.h5', mode='a') as f_extrin_opt:
            f_extrin_opt['/'+str(id)+'/rvec'][...] = rvecs.ravel()
            f_extrin_opt['/'+str(id)+'/tvec'][...] = tvecs

        rotM = cv2.Rodrigues(rvecs)[0]
        camerapos = -np.matrix(rotM).T * np.matrix(tvecs)
        print(id, ' (after):', np.array(camerapos).ravel())
"""