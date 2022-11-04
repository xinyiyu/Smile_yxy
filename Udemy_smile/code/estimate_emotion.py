import os, sys, time
threads = '4'
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

class EmonetDataset(Dataset):
    def __init__(self, img_files, transform=None):
        self.img_files = img_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        frame_id = os.path.splitext(os.path.basename(self.img_files[index]))[0]
        img = cv2.imread(self.img_files[index])
        if self.transform is not None:
            img = self.transform(img)
        return dict(id=frame_id, image=img)

def get_emotion_vai(img_files, dev):
    
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
    emonet_dataset = EmonetDataset(img_files, transform=trans_emonet)
    batch_size = 64 
    emonet_dataloader = DataLoader(emonet_dataset, batch_size=batch_size, shuffle=False)

    ### forward emonet
    emotions, vais = [], []
    for i, data in enumerate(emonet_dataloader):

        images = data['image'].to(dev)
        with torch.no_grad():
            out = emonet(images)
        emos = torch.softmax(out['expression'], axis=1).cpu().detach().numpy()  # (bs, 8)
        emos_df = pd.DataFrame(emos, columns=list(emotions_emonet.keys()))
        emos_df.insert(0, 'Frame_ID', data['id'])
        emotions.append(emos_df)
        vals = out['valence'].cpu().detach().numpy()  # (bs,)
        arous = out['arousal'].cpu().detach().numpy()  # (bs,)
        intens = np.sqrt(vals ** 2 + arous ** 2)
        vai_df = pd.DataFrame({'Frame_ID': data['id'], 'Valence': vals, 'Arousal': arous, 'Intensity': intens})
        vais.append(vai_df)
        if dev != 'cpu':
            torch.cuda.empty_cache()
    emotions_df = pd.concat(emotions)
    vais_df = pd.concat(vais)
    sort_index = emotions_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
    emotions_df = emotions_df.iloc[sort_index].reset_index(drop=True)
    sort_index = vais_df['Frame_ID'].apply(lambda x: x.split('_')[0].replace('frame','')).astype(int).argsort()
    vai_df = vais_df.iloc[sort_index].reset_index(drop=True)
    
    return emotions_df, vais_df


def main(video_name):
    
    video_dir = f'{video_root}/{video_name}'
    videos = sorted([os.path.splitext(x)[0] for x in os.listdir(video_dir)])
    print('Number of videos: ', len(videos))
    align_root = f'{category}/{video_name}/freq{freq}_aligned'
    score_root = f'{category}/{video_name}/freq{freq}_scores'
    assert os.path.exists(align_dir), 'Directory saving aligned face images does not exist!'
    assert os.path.exists(align_dir), 'Directory saving scores does not exist!'

    for idx, video in enumerate(videos):
        print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}][{idx}/{len(videos)}]")
        align_dir = os.path.join(align_root, video)
        score_dir = os.path.join(score_root, video)
        # get faces to be estimated
        info_df = pd.read_csv(os.path.join(score_dir, 'frame_info.csv'), sep='\t')
        frame_ids = info_df.loc[info_df['is_target'] == 1, 'Frame_ID'].values.tolist()
        img_files = [os.path.join(align_dir, f'{x}.jpg') for x in frame_ids]
        if len(img_files) > 0:
            if not os.path.exists(os.path.join(score_dir, 'target_emotion_vai.csv')):
                # estimate euler angles
                print(f'Video {video}: {len(img_files)} faces, estimate emotion.')
                t0 = time.time()
                emotions_df, vais_df = get_emotion_vai(img_files, 'cpu')
                merge_df = pd.merge(emotions_df, vais_df, on='Frame_ID')
                # save merge_df
                merge_df.to_csv(os.path.join(score_dir, 'target_emotion_vai.csv'), sep='\t', index=False)
                print(f'Video {video}: estimation emotions using {time.time() - t0:.2f}s.\n')
            else:
                print(f'Video {video}: Have finished estimation (emotion).\n')
        else:
            print(f'Video {video}: No face to be estimated (emotion)!\n')
            
if __name__ == '__main__':
    
    ## Specify these variables similar in 'detect_faces.py'
    freq = 5
    category = 'pd' # or 'fin'
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
        logging.error(e)
        
        
