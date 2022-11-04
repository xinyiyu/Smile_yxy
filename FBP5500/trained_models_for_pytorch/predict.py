import numpy as np
import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import Nets


class FBPDataset(Dataset):
    
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        
    def __len__(self):
        return len(glob.glob(os.path.join(self.root, '*.jpg')))
    
    def __getitem__(self, index):
        image_files = sorted([x.split('/')[1] for x in glob.glob(f'{self.root}/*.jpg')])
        image = io.imread(os.path.join(self.root, image_files[index]))
        if self.transform_image is not None:
            image = self.transform_image(image)
        return (image_files[index].split('.')[0], image)

def main():
    device = 'cuda'

    # net definition 
    net = Nets.AlexNet().to(device)

    model_dict = net.state_dict()
    pretrained_dict = torch.load('/home/xyubl/Smile/FBP5500/trained_models_for_pytorch/models/alexnet.pth', encoding='latin1')
    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    # evaluate
    net.eval()

    # loading data...
    root = '/home/xyubl/Smile/aligned'
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])  
    test_dataset = FBPDataset(root, transform)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=8, shuffle=False)

    scores = []
    t0 = time.time()
    for i, (img_id, img) in enumerate(test_loader):
        img = img.to(device) # (bs, 3, 224, 224)
        with torch.no_grad():
            score = net(img)
        score_df = pd.DataFrame(score.cpu().detach().numpy(), columns=['Beauty'])
        score_df.insert(0, 'seller_ID', img_id)
        scores.append(score_df)
    scores_df = pd.concat(scores)
    scores_df = scores_df.sort_values('seller_ID').reset_index(drop=True)
    print(f'Time FBP5500: {round(time.time() - t0, 2)}s')

    scores_df.to_csv('/home/xyubl/Smile/test_FBP.csv', sep='\t', index=False)

if __name__ == '__main__':
    main()
