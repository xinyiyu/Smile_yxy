### 5miles facial expression estimation
import os, sys, time, glob
threads = '8'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import reduce
from skimage import io

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from utils import *
from emonet.emonet.models import EmoNet

class EmonetDataset(Dataset):
    def __init__(self, root_path, transform_image=None):
        self.root_path = root_path
        self.transform_image = transform_image

    def __len__(self):
        return len(glob.glob(os.path.join(self.root_path, '*.jpg')))

    def __getitem__(self, index):
        image_files = sorted([x.split('/')[1] for x in glob.glob('aligned/*.jpg')])
        image = io.imread(os.path.join(self.root_path, image_files[index]))
        if self.transform_image is not None:
            image = self.transform_image(image)
        return dict(id=image_files[index].split('.')[0], image=image)
    
def emonet_predict(n_expression, emonet_dataloader, emotions_emonet):
    
    # load emonet
    emotions_emonet = eval(f'emotions_emonet_{n_expression}')
    emonet_state_path = f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))}/emonet/pretrained/emonet_{n_expression}.pth"
    emonet_state_dict = torch.load(str(emonet_state_path), map_location='cpu')
    emonet_state_dict = {k.replace('module.', ''): v for k, v in emonet_state_dict.items()}
    emonet = EmoNet(n_expression=n_expression).to(dev)
    emonet.load_state_dict(emonet_state_dict, strict=False)
    emonet.eval()
    
    scores = dict.fromkeys(['align_emos', 'align_vai'])
    emotions, vais = [], []
    # forward emonet
    t0 = time.time()
    for idx, data in enumerate(tqdm(emonet_dataloader)):

        images = data['image'].to(dev)
        with torch.no_grad():
            out = emonet(images)
        emos = torch.softmax(out['expression'], axis=1).cpu().detach().numpy()  # (bs, 8) or (bs, 5)
        emos_df = pd.DataFrame(emos, columns=list(emotions_emonet.keys()))
        emos_df.insert(0, 'seller_ID', data['id'])
        emotions.append(emos_df)
        vals = out['valence'].cpu().detach().numpy()  # (bs,)
        arous = out['arousal'].cpu().detach().numpy()  # (bs,)
        intens = np.sqrt(vals ** 2 + arous ** 2)
        vai_df = pd.DataFrame({'seller_ID': data['id'], 'Valence': vals, 'Arousal': arous, 'Intensity': intens})
        vais.append(vai_df)
        # print(f'{idx} / {len(emonet_dataset) // batch_size}')
    print(f'Time EmoNet: {round(time.time() - t0, 2)}s')
    if dev != 'cpu':
        torch.cuda.empty_cache()
    emotions_df = pd.concat(emotions)
    vais_df = pd.concat(vais)
    
    # sort dataframes
    emotions_df = emotions_df.sort_values('seller_ID').reset_index(drop=True)
    # rename emotions_df columns
    emotions_df.rename(columns=eval(f'emotions_emonet_{n_expression}'), inplace=True)
    vai_df = vais_df.sort_values('seller_ID').reset_index(drop=True)
    smile_df = pd.DataFrame({'seller_ID': emotions_df['seller_ID']})
    smile_df['smile'] = np.where(np.argmax(emotions_df.iloc[:,1:].values, axis=1) == 1, 1, 0)
    
    save_dfs = [smile_df, vais_df, emotions_df]
    save_df = reduce(lambda  left,right: pd.merge(left, right, on=['seller_ID'], how='outer'), save_dfs)

    return save_df

if __name__ == '__main__':
    
    align_dir = 'aligned' # folder saving aligned faces
    score_dir = 'scores' # folder for saving scores
    assert os.path.exists(align_dir), 'Directory saving aligned face images does not exist!'
    assert os.path.exists(score_dir), 'Directory saving scores does not exist!'
    
    # save as a dict with 8 keys
    dev = 'cuda'

    # emonet has two versions: 8 and 5 emotions, use 8-emotion version
    emotions_emonet_8 = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
                   4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt'}

    # setting
    batch_size = 64 if dev != 'cpu' else len(align_ids)
    n_workers = 4 if dev != 'cpu' else 0 # num_workers set to 4 or 8 fastest
    if dev != 'cpu':
        cudnn.benchmark = True

    # load and transform images
    trans_emonet = transforms.ToTensor()

    # data
    emonet_dataset = EmonetDataset(root_path=align_dir, transform_image=trans_emonet)
    emonet_dataloader = DataLoader(emonet_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    print(f'Number of aligned images for EmoNet: {len(emonet_dataset)}')
    
    # predict: 8 classes
    scores_8 = emonet_predict(8, emonet_dataloader, emotions_emonet_8)
    
    scores_8.to_csv(f'{score_dir}/seller_emotion.csv', sep='\t', index=False)

    