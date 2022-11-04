### 5miles facial expression estimation
import os, sys, time, shutil, glob
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
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from utils import *
from FBP5500.trained_models_for_pytorch import Nets

class FBPDataset(Dataset):
    
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        
    def __len__(self):
        return len(glob.glob(os.path.join(self.root, '*.jpg')))
    
    def __getitem__(self, index):
        image_files = sorted([x.split('/')[-1] for x in glob.glob(f'{self.root}/*.jpg')])
        image = Image.open(os.path.join(self.root, image_files[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return (image_files[index].split('.')[0], image)
    
if __name__ == '__main__':
    
    align_dir = 'aligned' # folder saving aligned faces
    score_dir = 'scores' # folder for saving scores
    assert os.path.exists(align_dir), 'Directory saving aligned face images does not exist!'
    assert os.path.exists(score_dir), 'Directory saving scores does not exist!'
    
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

    # loading data...
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  
    test_dataset = FBPDataset(align_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=8, shuffle=False)

    scores = []
    t0 = time.time()
    for i, (img_id, img) in enumerate(test_loader):
        img = img.to(device) # (bs, 3, 224, 224)
        with torch.no_grad():
            score = net(img)
        score_df = pd.DataFrame(score.cpu().detach().numpy(), columns=['beauty'])
        score_df.insert(0, 'seller_ID', img_id)
        scores.append(score_df)
    scores_df = pd.concat(scores)
    scores_df = scores_df.sort_values('seller_ID').reset_index(drop=True)
    print(f'Time FBP5500: {round(time.time() - t0, 2)}s')

    scores_df.to_csv(f'{score_dir}/seller_beauty.csv', sep='\t', index=False)
    