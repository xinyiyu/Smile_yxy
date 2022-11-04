import os, sys, time
threads = '8'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import cv2
from datetime import datetime

from utils import *
from multiprocessing import Pool

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from emonet.emonet.models import EmoNet

emotions_emonet = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
                   4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'}
device = 'cuda'

class EmonetDataset(Dataset):
    def __init__(self, img_files, transform=None, composite=['none', 'left', 'right']):
        self.img_files = img_files
        self.transform = transform
        self.composite = composite

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        frame_id = os.path.splitext(os.path.basename(self.img_files[index]))[0]
        img = cv2.imread(self.img_files[index])
        if self.composite == 'left':
            xcenter = int(img.shape[1] // 2)
            flipped = cv2.flip(img, 1)  # 1: horizontal, 0: vertical, -1: horizontal + vertical
            img = np.hstack((img[:, :xcenter, :], flipped[:, xcenter:, :]))
        elif self.composite == 'right':
            xcenter = int(img.shape[1] // 2)
            flipped = cv2.flip(img, 1)  # 1: horizontal, 0: vertical, -1: horizontal + vertical
            img = np.hstack((flipped[:, :xcenter, :], img[:, xcenter:, :]))
        if self.transform is not None:
            img = self.transform(img)
        return dict(id=frame_id, image=img)

def get_emotion(img_files, dev, composite=['none', 'left', 'right']):
    
    '''
    :param img_dir: directory containing aligned face images.
    :return emotions_df, vais_df: emotion and vai dataframe
    '''
    
    ### load emonet
    n_expression = 8
    n_workers = min(8, os.cpu_count()//len(device)+1) if dev != 'cpu' else 0 # num_workers set to 4 or 8 fastest
    if dev != 'cpu':
        cudnn.benchmark = True
        
    emonet_state_path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))}/emonet/pretrained/emonet_8.pth"
    emonet_state_dict = torch.load(str(emonet_state_path), map_location='cpu')
    emonet_state_dict = {k.replace('module.', ''): v for k, v in emonet_state_dict.items()}
    emonet = EmoNet(n_expression=n_expression).to(dev)
    emonet.load_state_dict(emonet_state_dict, strict=False)
    emonet.eval()

    ### dataset
    trans_emonet = transforms.ToTensor()
    emonet_dataset = EmonetDataset(img_files, transform=trans_emonet, composite=composite)
    batch_size = 64
    emonet_dataloader = DataLoader(emonet_dataset, batch_size=batch_size, shuffle=False)

    ### forward emonet
    emotions = []
    for i, data in enumerate(emonet_dataloader):

        images = data['image'].to(dev)
        with torch.no_grad():
            out = emonet(images)
        emos = torch.softmax(out['expression'], axis=1).cpu().detach().numpy()  # (bs, 8)
        emos_df = pd.DataFrame(emos, columns=list(emotions_emonet.keys()))
        emos_df.insert(0, 'Frame_ID', data['id'])
        emotions.append(emos_df)
        if dev != 'cpu':
            torch.cuda.empty_cache()
    emotions_df = pd.concat(emotions)
    sort_index = emotions_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
    emotions_df = emotions_df.iloc[sort_index].reset_index(drop=True)
    return emotions_df


def main():
    
    for video_name in video_folders:
        
        print(f'\n***** Processing {video_name} *****')
        start = time.time()
        
        video_dir = f'/home/xyubl/Downloads/{video_name}'
        assert os.path.isdir(video_root), 'Please specify the root saving videos!'
        
        videos = sorted([os.path.splitext(x)[0] for x in os.listdir(video_dir)])
        print('Number of videos: ', len(videos))
        align_root = f'{category}/{video_name}/freq{freq}_aligned'
        score_root = f'{category}/{video_name}/freq{freq}_scores'
        assert os.path.exists(align_dir), 'Directory saving aligned face images does not exist!'
        assert os.path.exists(score_dir), 'Directory saving scores does not exist!'

        for idx, video in enumerate(videos):
            print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}][{idx}/{len(videos)}]")
            align_dir = os.path.join(align_root, video)
            score_dir = os.path.join(score_root, video)
            # get faces to be estimated
            info_df = pd.read_csv(os.path.join(score_dir, 'frame_info.csv'), sep='\t')
            frame_ids = info_df.loc[info_df['is_target'] == 1, 'Frame_ID'].values.tolist()
            img_files = [os.path.join(align_dir, f'{x}.jpg') for x in frame_ids]
            if len(img_files) > 0:
                if not os.path.exists(os.path.join(score_dir, 'target_asymmetry.csv')):
                    # estimate euler angles
                    print(f'Video {video}: {len(img_files)} faces, estimate emotion.')
                    t0 = time.time()
                    left_df = get_emotion(img_files, device, 'left')
                    right_df = get_emotion(img_files, device, 'right')
                    asym_df = left_df.copy()
                    asym_df.iloc[:, 1:] = left_df.iloc[:, 1:] - right_df.iloc[:, 1:]
                    asym_df['asymmetry'] = np.linalg.norm(asym_df.iloc[:,1:], axis=1)
                    # save merge_df
                    asym_df.to_csv(os.path.join(score_dir, 'target_asymmetry.csv'), sep='\t', index=False)
                    print(f'Video {video}: estimation asymmetry using {time.time() - t0:.2f}s.\n')
                
        print(f'\n***** Finish {video_name}: {time.time() - start:.2f}s *****')

            
if __name__ == '__main__':
    
    freq = 5
    category = 'pd'
    if category == 'fin':
        video_folders = ['video_1-100', 'video_101-200', 'video_201-300', 'video_301-350', 'video_351-400', 'video_401-500', \
                         'video_501-580', 'video_581-660', 'video_661-740', 'video_741-820', 'video_821-900']
    elif category == 'pd':
        video_folders = ['pd_1-100', 'pd_101-200', 'pd_201-300', 'pd_301-400', 'pd_401-500',
                         'pd_501-600', 'pd_601-700', 'pd_701-800', 'pd_801-897']

    main()
        
        
