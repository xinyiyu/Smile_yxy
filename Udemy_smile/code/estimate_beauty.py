import os, sys, time, shutil
threads = '8'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import random
from PIL import Image

from utils import *

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from FBP5500.trained_models_for_pytorch import Nets

class FBPDataset(Dataset):
    
    def __init__(self, img_files, transform):
        self.img_files = img_files
        self.transform = transform
        
    def __len__(self):
        return len(img_files)
    
    def __getitem__(self, index):
        image = Image.open(self.img_files[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return (img_files[index].split('/')[-1].split('.')[0], image)
    
    
if __name__ == '__main__':
    
    device = 'cuda'

    # net definition 
    net = Nets.AlexNet().to(device)

    model_dict = net.state_dict()
    pretrained_dict = torch.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))}/FBP5500/trained_models_for_pytorch/models/alexnet.pth", encoding='latin1')
    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    # evaluate
    net.eval()
    
    # transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  


    # data
    category = 'pd'
    if category == 'fin':
        video_folders = ['video_1-100', 'video_101-200', 'video_201-300', 'video_301-350', 'video_351-400', 'video_401-500', \
                         'video_501-580', 'video_581-660', 'video_661-740', 'video_741-820', 'video_821-900']
    elif category == 'pd':
        video_folders = ['pd_1-100', 'pd_101-200', 'pd_201-300', 'pd_301-400', 'pd_401-500',
                         'pd_501-600', 'pd_601-700', 'pd_701-800', 'pd_801-897']
    
    freq = 5
    for video_name in video_folders:
        
        print(f'\n***** Processing {video_name} *****')
        start = time.time()
        
        video_dir = f'/home/xyubl/Downloads/{video_name}'
        assert os.path.isdir(video_root), 'Please specify the root saving videos!'
        
        videos = sorted([os.path.splitext(x)[0] for x in os.listdir(video_dir)])
        print(f'Number of videos in {video_name}: ', len(videos))
        align_root = f'{category}/{video_name}/freq{freq}_aligned'
        score_root = f'{category}/{video_name}/freq{freq}_scores'
        assert os.path.exists(align_dir), 'Directory saving aligned face images does not exist!'
        assert os.path.exists(score_dir), 'Directory saving scores does not exist!'
        
        for video in videos:
            align_dir = f'{align_root}/{video}'
            score_dir = f'{score_root}/{video}'
            if os.path.exists(f'{score_dir}/target_beauty.csv'):
                continue
            
            # get faces to be estimated
            if os.path.exists(os.path.join(score_dir, 'frame_info.csv')):
                info_df = pd.read_csv(os.path.join(score_dir, 'frame_info.csv'), sep='\t')
                frame_ids = info_df.loc[info_df['is_target'] == 1, 'Frame_ID'].values.tolist()
                img_files = [os.path.join(align_dir, f'{x}.jpg') for x in frame_ids]
                
                if len(img_files) > 0:
                    print(f'Video {video}: {len(img_files)} target images')
                    test_dataset = FBPDataset(img_files, transform)
                    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, shuffle=False)

                    scores = []
                    t0 = time.time()
                    for i, (img_id, img) in enumerate(test_loader):
                        img = img.to(device) # (bs, 3, 224, 224)
                        with torch.no_grad():
                            score = net(img)
                        score_df = pd.DataFrame(score.cpu().detach().numpy(), columns=['beauty'])
                        score_df.insert(0, 'Frame_ID', img_id)
                        scores.append(score_df)
                    scores_df = pd.concat(scores)
                    # print(scores_df.head())
                    # scores_df = scores_df.sort_values('Frame_ID', key=lambda x: int(x.values.split('_')[0].replace('frame', ''))).reset_index(drop=True)
                    print(f'Time {video_name}: {round(time.time() - t0, 2)}s')
                    scores_df.to_csv(f'{score_dir}/target_beauty.csv', sep='\t', index=False)
        print(f'\n***** Finish {video_name}: {time.time() - start:.2f}s *****')

    
    