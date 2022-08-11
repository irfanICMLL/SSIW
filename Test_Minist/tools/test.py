#!/usr/bin/python3

import os, sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_SPACE)
os.chdir(CODE_SPACE)

import argparse
import cv2
import logging
import numpy as np
import torch
import torch.nn.functional as F
import json

import utils.config as config
from utils.config import CfgNode
from utils.transforms_utils import get_imagenet_mean_std, normalize_img, pad_to_crop_sz, resize_by_scaled_short_side
import matplotlib.pyplot as plt
from utils.color_seg import color_seg

import glob
from PIL import Image
from utils.labels_dict import UNI_UID2UNAME, ALL_LABEL2ID, UNAME2EM_NAME
import torch.multiprocessing as mp
from utils.segformer import get_configured_segformer
from tqdm import tqdm 

def get_logger():
    """
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

logger = get_logger()


def get_parser() -> CfgNode:
    """
    TODO: add to library to avoid replication.
    """
    parser = argparse.ArgumentParser(description='Yvan Yin\'s Semantic Segmentation Model.')
    parser.add_argument('--root_dir', type=str, help='root dir for the data')
    parser.add_argument('--cam_id', type=str, help='camera ID')
    parser.add_argument('--img_folder', default='image_', type=str, help='the images folder name except the camera ID')
    parser.add_argument('--img_file_type', default='jpeg', type=str, help='the file type of images, such as jpeg, png, jpg...')

    parser.add_argument('--config', type=str, default='720_ss', help='config file')
    parser.add_argument('--gpus_num', type=int, default=2, help='number of gpus')
    parser.add_argument('--save_folder', type=str, default='ann/semantics', help='the folder for saving semantic masks')
    parser.add_argument('opts', help='see mseg_semantic/config/test/default_config_360.yaml for all options, model path should be passed in',
        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config_path = os.path.join('configs', f'{args.config}.yaml')
    args.config = config_path

    # test on samples
    if args.root_dir is None:
        args.root_dir = f'{CODE_SPACE}/test_imgs'
        args.cam_id='01'
        args.img_file_type = 'png'

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.root_dir = args.root_dir
    cfg.cam_id = args.cam_id
    cfg.img_folder = args.img_folder
    cfg.img_file_type = args.img_file_type
    cfg.gpus_num = args.gpus_num
    cfg.save_folder = args.save_folder
    return cfg


  
def get_prediction(embs, gt_embs_list):
    prediction = []
    logits = []
    B, _, _, _ = embs.shape
    for b in range(B):
        score = embs[b,...]
        score = score.unsqueeze(0)
        emb = gt_embs_list
        emb = emb / emb.norm(dim=1, keepdim=True)
        score = score / score.norm(dim=1, keepdim=True)
        score = score.permute(0, 2, 3, 1) @ emb.t()
        # [N, H, W, num_cls] You maybe need to remove the .t() based on the shape of your saved .npy
        score = score.permute(0, 3, 1, 2)  # [N, num_cls, H, W]
        prediction.append(score.max(1)[1])
        logits.append(score)
    if len(prediction) == 1:
        prediction = prediction[0]
        logit = logits[0]
    else:
        prediction = torch.cat(prediction, dim=0)
        logit = torch.cat(logits, dim=0)
    return logit

  
def single_scale_single_crop_cuda(model,
                      image: np.ndarray,
                      h: int, w: int, gt_embs_list,
                      args=None) -> np.ndarray:
    ori_h, ori_w, _ = image.shape
    mean, std = get_imagenet_mean_std()
    crop_h = (np.ceil((ori_h - 1) / 32) * 32).astype(np.int32)
    crop_w = (np.ceil((ori_w - 1) / 32) * 32).astype(np.int32)
    
    image, pad_h_half, pad_w_half = pad_to_crop_sz(image, crop_h, crop_w, mean)
    image_crop = torch.from_numpy(image.transpose((2, 0, 1))).float()
    normalize_img(image_crop, mean, std)
    image_crop = image_crop.unsqueeze(0).cuda()
    with torch.no_grad():
        emb, _, _ = model(inputs=image_crop, label_space=['universal'])
        logit = get_prediction(emb, gt_embs_list)
    logit_universal = F.softmax(logit * 100, dim=1).squeeze()

    # disregard predictions from padded portion of image
    prediction_crop = logit_universal[:, pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]

    # CHW -> HWC
    prediction_crop = prediction_crop.permute(1, 2, 0)
    prediction_crop = prediction_crop.data.cpu().numpy()

    # upsample or shrink predictions back down to scale=1.0
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

  
def single_scale_cuda(model,
                      image: np.ndarray,
                      h: int, w: int, gt_embs_list, stride_rate: float = 2/3,
                      args=None) -> np.ndarray:
    mean, std = get_imagenet_mean_std()
    crop_h = args.test_h
    crop_w = args.test_w
    ori_h, ori_w, _ = image.shape
    image, pad_h_half, pad_w_half = pad_to_crop_sz(image, crop_h, crop_w, mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)

    prediction_crop = torch.zeros((gt_embs_list.shape[0], new_h, new_w)).cuda()
    count_crop = torch.zeros((new_h, new_w)).cuda()
    # loop w/ sliding window, obtain start/end indices
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1

            image_crop = torch.from_numpy(image_crop.transpose((2, 0, 1))).float()
            normalize_img(image_crop, mean, std)
            image_crop = image_crop.unsqueeze(0)
            with torch.no_grad():
                emb, _, _ = model(inputs=image_crop, label_space=['universal'])
                logit = get_prediction(emb, gt_embs_list)
            logit_universal = F.softmax(logit * 100, dim=1)
            prediction_crop[:, s_h:e_h, s_w:e_w] += logit_universal.squeeze()

    prediction_crop /= count_crop.unsqueeze(0)
    # disregard predictions from padded portion of image
    prediction_crop = prediction_crop[:, pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]

    # CHW -> HWC
    prediction_crop = prediction_crop.permute(1,2,0)
    prediction_crop = prediction_crop.data.cpu().numpy()

    # upsample or shrink predictions back down to scale=1.0
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction

  
def do_test(args, local_rank):
    imgs_on_devices = organize_images(args, local_rank)
    model = get_configured_segformer(args.num_model_classes,
                                     criterion=None,
                                     load_imagenet_model=False)
    model.eval()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[local_rank,],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)

    ckpt_path = args.model_path
    checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
    model.load_state_dict(ckpt_filter, strict=True)

    gt_embs_list = torch.tensor(np.load(args.emb_path)).cuda().float()
    id_to_label = UNI_UID2UNAME

    test_single(args, imgs_on_devices, local_rank, model, gt_embs_list)
  
def test_single(args, imgs_list, local_rank, model, gt_embs_list):
    for i, rgb_path in tqdm(enumerate(imgs_list)):
        save_path = os.path.join(args.root_dir, args.save_folder, os.path.basename(rgb_path))
        save_path = os.path.splitext(save_path)[0] + '.png'
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        rgb = cv2.imread(rgb_path, -1)[:, :, ::-1]
        image_resized = resize_by_scaled_short_side(rgb, args.base_size, 1)
        h, w, _ = rgb.shape
        
        if args.single_scale_single_crop:
            out_logit = single_scale_single_crop_cuda(model, image_resized, h, w, gt_embs_list=gt_embs_list, args=args)
        elif args.single_scale_multi_crop:
            out_logit = single_scale_cuda(model, image_resized, h, w, gt_embs_list=gt_embs_list, args=args)
                                      
        prediction = out_logit.argmax(axis=-1).squeeze()
        probs = out_logit.max(axis=-1).squeeze()
        high_prob_mask = probs > 0.5
        
        mask = high_prob_mask
        prediction[~mask] = 255
        
        pred_color = color_seg(prediction)
        vis_seg = visual_segments(pred_color, rgb)
        
        vis_seg.save(os.path.splitext(save_path)[0] + '_vis.png')
        cv2.imwrite(save_path, prediction.astype(np.uint8))
        
def visual_segments(segments, rgb):
    seg = Image.fromarray(segments)
    rgb = Image.fromarray(rgb)

    seg1 = seg.convert('RGBA')
    rgb1 = rgb.convert('RGBA')

    vis_seg = Image.blend(rgb1, seg1, 0.8)
    return vis_seg
  
def organize_images(args, local_rank):
    imgs_dir = args.root_dir
    imgs_list = glob.glob(imgs_dir + f'/*.{args.img_file_type}')
    imgs_list.sort()
    num_devices = args.gpus_num

    imgs_on_device = imgs_list[local_rank::num_devices]
    return imgs_on_device

def main_worker(local_rank: int, cfg: dict):
    if cfg.distributed:
        global_rank = loca_rank
        world_size = cfg.gpus_num
        
        torch.cuda.set_device(global_rank)
        dist.init_process_group(backend="nccl",
                            init_method=cfg.dist_url,
                            world_size=world_size,
                            rank=global_rank,)
    do_test(cfg, local_rank)
if __name__ == '__main__':
    args = get_parser()
    logger.info(args)
    
    dist_url = 'tcp://127.0.0.1:6769'
    dist_url = dist_url[:-2] + str(os.getpid() % 100).zfill(2)
    args.dist_url = dist_url
    
    num_gpus = torch.cuda.device_count()
    if num_gpus != args.gpus_num:
        raise RuntimeError('The set gpus number cannot match the detected gpus number. Please check or set CUDA_VISIBLE_DEVICES')
    
    if num_gpus > 1:
        args.distributed = True
    else:
        args.distributed = False
    
    save_path = os.path.join(args.root_dir, args.save_folder, 'id2labels.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(UNI_UID2UNAME, f)
    
    if not args.distributed:
        main_worker(0, args)
    else:
        mp.spawn(main_worker, nprocs=args.gpus_num, args=(args, ))
