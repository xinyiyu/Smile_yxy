"""Test Demo for Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
import os, sys, glob, time
threads = '4'
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cv2
import torch
from torchvision import transforms
import skvideo.io
from PIL import Image
import numpy as np
from VSFA import VSFA
from VSFA.CNNfeatures import get_features
from argparse import ArgumentParser
from multiprocessing import Pool
from datetime import datetime
from tqdm import tqdm

def vqa(video):
    
    video_name = video.split('/')[-1].split('.')[0]
    
    start = time.time()
    
    vid = cv2.VideoCapture(video)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < args.num_frames:
        num_frames = frame_count
        exclude_frames = 0
    else:
        num_frames = args.num_frames
        exclude_frames = args.exclude_frames
    
    # data preparation
    assert args.video_format == 'YUV420' or args.video_format == 'RGB'
    if args.video_format == 'YUV420':
        video_data = skvideo.io.vread(video, args.video_height, args.video_width, num_frames, inputdict={'-pix_fmt': 'yuvj420p'})
    else:
        video_data = skvideo.io.vread(video, num_frames=num_frames)
    
    video_data = video_data[exclude_frames:num_frames]
    video_length = video_data.shape[0]
    video_channel = video_data.shape[3]
    video_height = video_data.shape[1]
    video_width = video_data.shape[2]
    transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for frame_idx in range(video_length):
        frame = video_data[frame_idx]
        frame = Image.fromarray(frame)
        frame = transform(frame)
        transformed_video[frame_idx] = frame

    # print('Video length: {}'.format(transformed_video.shape[0]))

    # feature extraction
    features = get_features(transformed_video, frame_batch_size=args.frame_batch_size, device=device)
    features = torch.unsqueeze(features, 0)  # batch size 1

    with torch.no_grad():
        input_length = features.shape[1] * torch.ones(1, 1)
        outputs = model(features, input_length)
        y_pred = outputs[0][0].to('cpu').numpy()
        # print("Predicted quality: {}".format(y_pred))

    end = time.time()

    # print('Time: {} s'.format(end-start))
    
    with open(f'{category}/video_quality.txt', 'a') as f:
        f.write(f"{video_name}\t{y_pred}\n")
        
    return y_pred


if __name__ == "__main__":
    
    parser = ArgumentParser(description='Test Demo of VSFA')
    parser.add_argument('--model_path', default=f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))}/VSFA/models/VSFA.pt", type=str,
                        help='model path (default: models/VSFA.pt)')
    parser.add_argument('--video_format', default='RGB', type=str,
                        help='video format: RGB or YUV420 (default: RGB)')
    parser.add_argument('--video_width', type=int, default=None,
                        help='video width')
    parser.add_argument('--video_height', type=int, default=None,
                        help='video height')
    parser.add_argument('--exclude_frames', type=int, default=150,
                        help='Exclude first exclude_frames frames')
    parser.add_argument('--num_frames', type=int, default=600,
                        help='Number of frames for evaluation')
    parser.add_argument('--frame_batch_size', type=int, default=16,
                        help='frame batch size for feature extraction')
    args = parser.parse_args()

    multiprocess = False
    device = 'cuda'

    video_root = '/home/xyubl/Downloads'
    assert os.path.isdir(video_root), 'Please specify the root saving videos!'
    
    category = 'pd' # 'fin' or 'pd'
    if category == 'fin':
        video_folders = ['video_1-100', 'video_101-200', 'video_201-300', 'video_301-350', 'video_351-400', 'video_401-500', \
                     'video_501-580', 'video_581-660', 'video_661-740', 'video_741-820', 'video_821-900']
    elif category == 'pd':
        video_folders = ['pd_1-100', 'pd_101-200', 'pd_201-300', 'pd_301-400', 'pd_401-500',
                         'pd_501-600', 'pd_601-700', 'pd_701-800', 'pd_801-897']
        
    if os.path.exists(f'{category}/video_quality.txt'):
        with open(f'{category}/video_quality.txt', 'r') as f:
            existing_results = f.readlines()
        existing_video_names = [x.split('\t')[0] for x in existing_results]
        last_video_name = existing_video_names[-1]
        print(f'Processed {len(existing_video_names)} videos, continue from video {last_video_name}')
        # os.remove(f'Udemy_smile/{category}/video_quality.txt')
    
    videos = []
    for video_folder in video_folders:
        videos.extend(sorted(glob.glob(f'{video_root}/{video_folder}/*.mp4')))
    video_names = [x.split('/')[-1].split('.')[0] for x in videos]
    if 'existing_results' in globals():
        video_names = sorted(list(set(video_names) - set(existing_video_names)))
        videos = [x for x in videos if x.split('/')[-1].split('.')[0] in video_names]
        print(f'Remain {len(videos)} videos')
    else:
        print(f'Total number of videos in category {category}: {len(videos)}')
    
    # quality prediction using VSFA
    model = VSFA()
    model.load_state_dict(torch.load(args.model_path))  #
    model.to(device)
    model.eval()
    
    if multiprocess:
        n = 20
        starts = list(range(0, len(videos), n))
        for start in starts:
            cur_videos = videos[start:(start+n)]
            inp_args = [(x,) for x in cur_videos]
            print(f"***** [{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Processing videos {start}-{min(start+n, len(videos))} *****")
            try:
                with Pool(len(inp_args)) as pool:
                    pool.starmap(vqa, inp_args)
                    pool.close()
                    pool.join()
            except Exception as e:
                print(e)
                logging.error(e)
    else:
        stream = tqdm(videos)
        for _, video in enumerate(stream):
            # print(f"[{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}] Processing video {video.split('/')[-1].split('.')[0]}")
            pred_quality = vqa(video)
            stream.set_description(f"Video {video.split('/')[-1].split('.')[0]}: {pred_quality:.3f}")
        
