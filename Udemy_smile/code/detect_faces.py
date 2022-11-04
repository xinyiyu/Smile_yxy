import os
threads = '4'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
import sys
import time
import pickle
import shutil
import argparse
import traceback
import numpy as np
import pandas as pd
import random
import cv2
import dlib
from datetime import datetime
from multiprocessing import Pool

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from utils import *

def extract_detect_align(video_file, img_dir, freq):
    
    
    '''
    Extract frames, detecta and align faces.
    
    :param video_file: the path of video
    :param img_dir: directory for saving the aligned faces
    :param freq: number of frames per second, default is 1. If freq=-1, extract all frames.
    :return video_dict: a dict saving info of each frame
    '''
    
    if img_dir and not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    # record (frames with 1 face): time, face location, landmarks
    video_id = os.path.splitext(os.path.basename(video_file))[0]
    video_dict = {}
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
    duration = round(frame_count/fps, 2)
    print(f'Video {video_id}: fps {fps}, frames {frame_count}, duration {duration}s.')
    count = -1
    success = True
    t0 = time.time()
    while success:
        ### extract frame
        success, image = cap.read()
        # milisecond at current time point
        if success:
            count += 1
            if freq > 0 and count % int(fps/freq) != 0:
                continue
            msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                        
            ### recognize face
            # face located in the rectangle (0.2-0.3s): list of dlib rectangles
            face_locations = face_detector(image)
            if len(face_locations) == 1:
                face_location = face_locations[0]
                # coordinates of 68 landmarks (0.004s): list of dlib full_object_detection objects
                landmark = pose_predictor_68_point(image, face_location)
                # face encodings (0.6-0.8s): list of (128,) arrays
                encoding = np.array(face_encoder.compute_face_descriptor(image, landmark))
                # record
                save_locations = rect_to_css(face_location) # convert rectangle to (top, right, bottom, left)
                save_2d_landmarks = landmarks_to_np(landmark) # convert dlib full_object_detection object to (68, 2) np.ndarray

                ### align recognized face
                aligned_face = face_align(image, landmark)
                
                ### record
                frame_id = f'frame{count}_msec{round(count*1000/fps)}'
                cur_frame = {}
                cur_frame['location'] = face_location
                cur_frame['2d_landmark'] = landmark
                cur_frame['encoding'] = encoding
                video_dict[frame_id] = cur_frame
                
                ### save aligned face
                cv2.imwrite(os.path.join(img_dir, f'frame{count}_msec{round(count*1000/fps)}.jpg'), aligned_face)

    cap.release()
    
    print(f'Video {video_id}: detect and aligned faces using {time.time() - t0:.2f}s.')
    
    return video_dict

def recognize_face(video_dict, min_dist=0.6, cluster_n=2):
    
    '''
    Recognize teachers' faces.
    
    :param video_dict: dict saving info of the video
    :return video_dict: updated video_dict with face info
    '''
    
    tot_num_faces = len(list(video_dict))
    
    ### K-means clustering: select reference encodings
    known_encodings, all_encodings = [], []
    for k, v in video_dict.items():
        all_encodings.append(v['encoding'])
    X = np.array(all_encodings)
    kmeans = KMeans(n_clusters=2).fit(X)
    y = kmeans.labels_
    large_id = 1 if sum(y) > len(y) / 2 else 0
    small_id = 1 - large_id
    mean_encoding = np.mean(X, axis=0)
    mean_encoding_l = np.mean(X[np.where(y == large_id)[0]], axis=0)
    mean_encoding_s = np.mean(X[np.where(y == small_id)[0]], axis=0)
    known_encodings.extend(X[np.argsort(np.linalg.norm(X[np.where(y == large_id)[0]] - mean_encoding_s, axis=1))[:cluster_n].tolist()])

    X_l = X[np.where(y == large_id)[0], :]
    X_s = X[np.where(y == small_id)[0], :]
    X_cluster = np.vstack([X_l, X_s])
    dist_cluster = euclidean_distances(X_cluster, X_cluster)
    min_between_dist_cluster = np.min(dist_cluster[X_l.shape[0]:, :X_l.shape[0]]) if X_l.shape[0] < X.shape[0] else 0
    # only use larger cluster if minimum distance between two clusters is greater than min_dist
    if min_between_dist_cluster <= min_dist:
        known_encodings.append(X[np.argmin(np.linalg.norm(X - mean_encoding, axis=1))])
        known_encodings.extend(X[np.argsort(np.linalg.norm(X[np.where(y == small_id)[0]] - mean_encoding_l, axis=1))[:cluster_n].tolist()])
    known_encodings = np.array(known_encodings)
    print('Number of reference faces: ', len(known_encodings))

    ### recognize teacher's faces
    encoding_list = [] 
    n = 0
    for i, (k, v) in enumerate(video_dict.items()):

        closest_dist = np.min(np.linalg.norm(known_encodings - v['encoding'], axis=1))
        if closest_dist <= min_dist:
            v['is_target'] = 1
            n += 1
        else:
            v['is_target'] = 0
    print(f'Target faces / total faces: {n} / {tot_num_faces}.')
    
    return video_dict
    
def main(video_name):
    
    video_dir = f'{video_root}/{video_name}'
    videos = sorted([os.path.splitext(x)[0] for x in os.listdir(video_dir)])
    print('Number of videos: ', len(videos))
    align_root = f'{category}/{video_name}/freq{freq}_aligned'
    score_root = f'{category}/{video_name}/freq{freq}_scores'
    os.makedirs(align_root, exist_ok=True)
    os.makedirs(score_root, exist_ok=True)
    
    for idx, video in enumerate(videos):
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}][{idx}/{len(videos)}]")
        video_file = os.path.join(video_dir, f'{video}.mp4')
        align_dir = os.path.join(align_root, video)
        score_dir = os.path.join(score_root, video)
        os.makedirs(align_dir, exist_ok=True)
        os.makedirs(score_dir, exist_ok=True)
        # get faces in the video
        video_dict = extract_detect_align(video_file, align_dir, freq=freq)
        # recognize if each face is target
        if len(list(video_dict)) < 10:
            print(f'Video {video}: Too few faces detected in the video, will not use this video!')
            for k, v in video_dict.items():
                v['is_target'] = 0
        else:
            video_dict = recognize_face(video_dict)
        # create info_df
        info_df = pd.DataFrame(columns=['Frame_ID', 'is_target'])
        for k, v in video_dict.items():
            info_df.loc[len(info_df)] = [k, v['is_target']]
        sort_index = info_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
        info_df = info_df.iloc[sort_index].reset_index(drop=True)
        # save video_dict and info_df
        with open(os.path.join(score_dir, 'frame_info_2d_68pts_encodings.pkl'), 'wb') as f:
            pickle.dump(video_dict, f)
        info_df.to_csv(os.path.join(score_dir, 'frame_info.csv'), sep='\t', index=False)
    
    
if __name__ == '__main__':
    

    ## Specify how many frames to be extracted per second, e.g freq=1, freq=5
    freq = 5
    ## Specify the course category, e.g 'fin' for finance, 'pd' for personal development
    category = 'pd'
    ## Specify the video folders to be processed in parallel, and the root containing these folders
    inp_args = [('pd_1-100',), ('pd_101-200',), ('pd_201-300',), ('pd_301-400',), ('pd_401-500',),
                ('pd_501-600',), ('pd_601-700',), ('pd_701-800',), ('pd_801-897',)]
    video_root = '/home/xyubl/Downloads'
    assert os.path.isdir(video_root), 'Please specify the root saving videos!'
    
    try:
        with Pool(int(threads) * len(inp_args)) as pool:
            pool.starmap(main, inp_args)
            pool.close()
            pool.join()
    except Exception as e:
        print(e)


    
