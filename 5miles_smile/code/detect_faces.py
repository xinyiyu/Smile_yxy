### 5miles detect faces
import os
import sys
import time
import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import *

if __name__ == '__main__':
    
    image_dir = 'seller_pictures' # folder saving seller pictures
    assert os.path.exists(image_dir), 'Directory saving seller pictures does not exist!'
    align_dir = 'aligned' # folder for saving aligned faces
    score_dir = 'scores' # folder for saving scores

    images = os.listdir('seller_pictures')
    image_id = sorted([x.split('.')[0] for x in images if len(x.split('.')[0]) > 0])
    os.makedirs(align_dir, exist_ok=True)
    os.makedirs(score_dir, exist_ok=True)
    face_dict = dict.fromkeys(image_id)
    face_df = pd.DataFrame(columns=['seller_ID', 'num_faces', 'face_proximity'])
    face_df.seller_ID = image_id
    
    t0 = time.time()
    for i, image_file in enumerate(tqdm(image_id)):
        image_info = dict.fromkeys(['num_faces', 'locations', '2d_landmarks', 'face_proximity'])
        image = cv2.imread(os.path.join(image_dir, f'{image_file}.jpg'))
        # detect face
        face_locations = face_detector(image) # list of dlib rectangles
        landmarks = [pose_predictor_68_point(image, face_location) for face_location in face_locations] # list of dlib full_object_detection objects
        # align: only one face in the photo
        if len(face_locations) == 1:
            aligned_face = face_align(image, landmarks[0])
            cv2.imwrite(os.path.join(align_dir, f'{image_file}.jpg'), aligned_face)
            # add face proximity
            h = face_locations[0].bottom() - face_locations[0].top()
            w = face_locations[0].right() - face_locations[0].left()
            face_area = h * w
            image_area = image.shape[0] * image.shape[1]
            face_proximity = min(round(face_area / image_area, 2), 1)
        else:
            face_proximity = None
        # record
        save_locations = [rect_to_css(x) for x in face_locations] # convert rectangle to (top, right, bottom, left)
        save_2d_landmarks = [landmarks_to_np(x) for x in landmarks] # convert dlib full_object_detection object to (68, 2) np.ndarray
        image_info['num_faces'], image_info['locations'], image_info['2d_landmarks'], image_info['face_proximity'] = \
            len(face_locations), save_locations, save_2d_landmarks, face_proximity
        face_df.iloc[i, 1] = len(face_locations)
        face_df.iloc[i, 2] = face_proximity
        face_dict[image_file] = image_info
        
    print(f'Time face detection: {round(time.time() - t0, 2)}')
    # save image info
    face_df.to_csv(f'{score_dir}/seller_pictures_info.csv', sep='\t', index=False)

