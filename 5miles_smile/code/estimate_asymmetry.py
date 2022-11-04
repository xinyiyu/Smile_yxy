import os, sys, time, glob
threads = '8'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from emonet.emonet.models import EmoNet
from utils import *

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
    for i, data in enumerate(tqdm(emonet_dataloader)):

        images = data['image'].to(dev)
        with torch.no_grad():
            out = emonet(images)
        emos = torch.softmax(out['expression'], axis=1).cpu().detach().numpy()  # (bs, 8)
        emos_df = pd.DataFrame(emos, columns=list(emotions_emonet.keys()))
        emos_df.insert(0, 'seller_ID', data['id'])
        emotions.append(emos_df)
        if dev != 'cpu':
            torch.cuda.empty_cache()
    emotions_df = pd.concat(emotions)
    emotions_df = emotions_df.sort_values('seller_ID').reset_index(drop=True)
    return emotions_df


if __name__ == '__main__':
    
    align_dir = 'aligned' # folder saving aligned faces
    score_dir = 'scores' # folder for saving scores
    assert os.path.exists(align_dir), 'Directory saving aligned face images does not exist!'
    assert os.path.exists(score_dir), 'Directory saving scores does not exist!'
    
    device = 'cuda'

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  

    emotions_emonet = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
                       4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'}
    
    img_files = glob.glob(f'{align_dir}/*.jpg')

    left_df = get_emotion(img_files, device, 'left')
    right_df = get_emotion(img_files, device, 'right')
    asym_df = left_df.copy()
    asym_df.iloc[:, 1:] = left_df.iloc[:, 1:] - right_df.iloc[:, 1:]
    asym_df['asymmetry'] = np.linalg.norm(asym_df.iloc[:,1:], axis=1)
    
    asym_df.to_csv(f'{score_dir}/seller_asymmetry.csv', sep='\t', index=False)
