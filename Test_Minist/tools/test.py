import argparse
import ast
import glob
import logging
import os
import sys
from pathlib import Path
from random import sample
from typing import Union, List, Dict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from segmentation_models_pytorch.losses import JaccardLoss,DiceLoss,SoftCrossEntropyLoss,FocalLoss
from tqdm import tqdm
import torchvision.transforms as transforms
import Test_Minist.utils.config as config
from Test_Minist.utils.color_seg import color_seg
from Test_Minist.utils.config import CfgNode
from Test_Minist.utils.get_class_emb import create_embs_from_names
from Test_Minist.utils.labels_dict import Background_Def
from Test_Minist.utils.labels_dict import UNI_UID2UNAME
from Test_Minist.utils.segformer import get_configured_segformer
from Test_Minist.utils.transforms_utils import get_imagenet_mean_std, normalize_img, pad_to_crop_sz, \
    resize_by_scaled_short_side

CODE_SPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_SPACE)
os.chdir(CODE_SPACE)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()


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
    parser.add_argument('--img_file_type', default='jpeg', type=str,
                        help='the file type of images, such as jpeg, png, jpg...')

    parser.add_argument('--config', type=str, default='720_ss', help='config file')
    parser.add_argument('--gpus_num', type=int, default=1, help='number of gpus')
    parser.add_argument('--save_folder', type=str, default='ann/semantics', help='the folder for saving semantic masks')

    parser.add_argument('--user_label', nargs='*', help='the label user identified for semantic segmentation')
    parser.add_argument('--new_definitions', type=ast.literal_eval, help='new label definitions identified by user')
    parser.add_argument('opts',
                        help='see mseg_semantic/config/test/default_config_360.yaml for all options, models path should be passed in',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config_path = os.path.join('configs', f'{args.config}.yaml')
    args.config = config_path

    # test on samples
    if args.root_dir is None:
        args.root_dir = f'{CODE_SPACE}/test_imgs'
        args.cam_id = '01'
        args.img_file_type = 'png'

    if args.user_label:
        args.user_label = [i for i in args.user_label]

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.root_dir = args.root_dir
    cfg.cam_id = args.cam_id
    cfg.img_folder = args.img_folder
    cfg.img_file_type = args.img_file_type
    cfg.gpus_num = args.gpus_num
    cfg.save_folder = args.save_folder

    cfg.user_label = args.user_label
    cfg.new_definitions = args.new_definitions
    return cfg


def get_prediction(embs: torch.Tensor, gt_embs_list: torch.Tensor) -> torch.Tensor:
    prediction = []
    logits = []
    B, _, _, _ = embs.shape
    for b in range(B):
        score = embs[b, ...]
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
        # prediction = prediction[0]
        logit = logits[0]
    else:
        # prediction = torch.cat(prediction, dim=0)
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


def single_scale_single_crop_cuda_train(model,
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
    model.train()
    with torch.set_grad_enabled(True):
        emb, _, _ = model(inputs=image_crop, label_space=['universal'])
    logit = get_prediction(emb, gt_embs_list)
    logit_universal = F.softmax(logit * 100, dim=1).squeeze()

    # disregard predictions from padded portion of image
    prediction_crop = logit_universal[:, pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    logit_for_loss = prediction_crop.unsqueeze(0)

    # CHW -> HWC
    prediction_crop = prediction_crop.permute(1, 2, 0)
    prediction_crop = prediction_crop.data.cpu().numpy()

    # upsample or shrink predictions back down to scale=1.0
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction, logit_for_loss


def single_scale_cuda(model,
                      image: np.ndarray,
                      h: int, w: int, gt_embs_list, stride_rate: float = 2 / 3,
                      args=None) -> np.ndarray:
    mean, std = get_imagenet_mean_std()
    crop_h = args.test_h
    crop_w = args.test_w
    ori_h, ori_w, _ = image.shape
    image, pad_h_half, pad_w_half = pad_to_crop_sz(image, crop_h, crop_w, mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)

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
    prediction_crop = prediction_crop[:, pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]

    # CHW -> HWC
    prediction_crop = prediction_crop.permute(1, 2, 0)
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
                                                          device_ids=[local_rank, ],
                                                          output_device=local_rank,
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)

    ckpt_path = args.model_path
    checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
    model.load_state_dict(ckpt_filter, strict=False)

    if args.user_label:
        if args.new_definitions:
            gt_embs_list = create_embs_from_names(args.user_label, args.new_definitions).float()
        else:
            gt_embs_list = create_embs_from_names(args.user_label).float()
        args.id_to_label = {i: v for i, v in enumerate(args.user_label)}
        args.id_to_label[255] = 'unlabel'
    else:
        gt_embs_list = torch.tensor(np.load(args.emb_path)).cuda().float()
        args.id_to_label = UNI_UID2UNAME
        if args.new_definitions:
            new_labels = [label for label in args.new_definitions.keys()]
            new_embs = create_embs_from_names(new_labels, args.new_definitions).float()
            if len(new_labels) == 1: new_embs = new_embs.unsqueeze(0)
            gt_embs_list = torch.cat([gt_embs_list, new_embs], dim=0)
            nums = len(args.id_to_label)
            for i in range(len(new_labels)):
                args.id_to_label[nums + i - 1] = new_labels[i]

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

        prediction = out_logit.argmax(axis=-1).squeeze()  # (h, w)
        probs = out_logit.max(axis=-1).squeeze()  # (h, w)
        high_prob_mask = probs > 0.5

        mask = high_prob_mask
        prediction[~mask] = 255

        labels = np.unique(prediction)
        label_names = {i: args.id_to_label[i] for i in labels}
        # label_names = [id_to_label[i] for i in labels]
        print()
        print(f'label_names for img{i + 1}', label_names)

        # # change to initial color
        # if not args.new_definitions:
        #     UNAME = [v for v in UNI_UID2UNAME.values()]
        #     if args.user_label:
        #         if 255 in labels: labels = np.delete(labels, np.where(labels == 255))
        #         for label in labels:  # 0~n的数字
        #             mask = (prediction == label)
        #             prediction[mask] = UNAME.index(args.user_label[label])

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
        global_rank = local_rank
        world_size = cfg.gpus_num

        torch.cuda.set_device(global_rank)
        dist.init_process_group(backend="nccl",
                                init_method=cfg.dist_url,
                                world_size=world_size,
                                rank=global_rank, )
    do_test(cfg, local_rank)


class Dataset:
    pass


class CMPFacade(Dataset):

    def __init__(self, data_dir: str = "/home/amirhossein/PycharmProjects/SSIW/CMP_facade_DB_base/base",
                 transform: List = None, resize_base_size=512, transforms=None):
        if transform:
            self.transform = transforms.Compose(transform)
        else:
            mean, std = get_imagenet_mean_std()
            img_height = 1024
            img_width = 543
            transforms_ = [
                transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
                transforms.RandomCrop((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
            self.transform = transforms.Compose(transforms_)
        self.img_files = sorted(glob.glob(data_dir + "/*.jpg"))
        self.annotations = sorted(glob.glob(data_dir + "/*.png"))
        self.color_dict = Image.open(self.annotations[0]).palette.colors
        self.resize_base_size = resize_base_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index) -> Dict:
        rgb_img = cv2.imread(self.img_files[index], -1)[:, :, ::-1]  ##convert bgr to rgb
        anot_img = cv2.imread(self.annotations[index], cv2.IMREAD_COLOR)
        img_height = 1024
        img_width = 543
        anot_img = cv2.resize(anot_img, (img_width, img_height))
        rgb_img = cv2.resize(rgb_img, (img_width, img_height))
        anot_img = self.anot_preproces(anot_img, self.color_dict)
        if self.transform:
            rgb_img = self.transform(Image.fromarray(rgb_img))
        return dict(img=rgb_img, mask=anot_img)

    @staticmethod
    def anot_preproces(anot_cv: np.ndarray, color_dict: Dict) -> Image.Image:
        '''
        convert each mask of shape [H,W,C=3]->[H,W,C=N]
        N: is number of categories
        '''
        output_mask = []
        for label in color_dict.keys():
            # cmap = np.all(np.equal(img.astype('int'), label), axis=-1)
            # cmap=np.where(np.all(np.equal(img, label), axis=-1),color_dict[label],0)
            new_mask = np.zeros(anot_cv.shape[:-1], dtype=np.int)
            new_mask[(anot_cv == np.array(label)).all(axis=2)] = 1
            output_mask.append(new_mask)

        output_mask = np.stack(output_mask, axis=-1)

        assert len(np.unique(output_mask)) == 2

        return np.transpose(output_mask, (2, 0, 1))


def anot_preproces(anot_cv: np.ndarray) -> np.ndarray:
    '''
    convert each mask of shape [H,W,C=3]->[H,W,C=N]
    N: is number of categories
    '''
    data_dir = '/home/amirhossein/PycharmProjects/SSIW/CMP_facade_DB_base/base'
    color_dict = Image.open(sorted(glob.glob(data_dir + "/*.png"))[0]).palette.colors
    output_mask = []
    for label in color_dict.keys():
        new_mask = np.zeros(anot_cv.shape[:-1], dtype=np.int)
        new_mask[(anot_cv == np.array(label)).all(axis=2)] = 1
        output_mask.append(new_mask)

    output_mask = np.stack(output_mask[1:], axis=-1)

    assert len(np.unique(output_mask)) == 2

    return np.transpose(output_mask, (2, 0, 1))


def train(data_dir: Union[str, Path] = '/home/amirhossein/PycharmProjects/SSIW/CMP_facade_DB_base/base'):
    imgs_train_list = sorted(glob.glob(data_dir + "/*.jpg"))[50:]
    anots_train_list = sorted(glob.glob(data_dir + "/*.png"))[50:]
    imgs_test_list = sorted(glob.glob(data_dir + "/*.jpg"))[:50]
    anots_test_list = sorted(glob.glob(data_dir + "/*.png"))[:50]
    num_model_classes: int = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_configured_segformer(num_model_classes,
                                     criterion=None,
                                     load_imagenet_model=False)
    model = torch.nn.DataParallel(model)
    ckpt_path = '/home/amirhossein/PycharmProjects/SSIW/Test_Minist/model/segformer_7data.pth'
    checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
    model.load_state_dict(ckpt_filter, strict=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    #I used two loss function for evaluation
    criterion = JaccardLoss(mode='multilabel', from_logits=False)
    # criterion = DiceLoss(mode='multilabel', from_logits=False)
    EPOCH = 5
    batch_size = 6
    epoch_loss = []
    all_loss = []
    best_iou_score = 0
    for j in range(EPOCH):
        loss_value = 0
        iou_value = 0
        f1_score_value = 0
        losses = []
        f1_score_test = []
        iou_score_test = []
        for i, (rgb_path, anot_path) in tqdm(
                enumerate(zip(imgs_train_list, anots_train_list))):
            # save_path = os.path.join(args.root_dir, args.save_folder, os.path.basename(rgb_path))
            # save_path = os.path.splitext(save_path)[0] + '.png'

            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            base_size = 256  ## it is according to my gpu power
            anot_img = cv2.imread(anot_path, cv2.IMREAD_COLOR)
            anot_img_resized = resize_by_scaled_short_side(anot_img, base_size, 1)
            ## convert anotation to acceptable format
            anot_img_modif = anot_preproces(anot_img_resized)  ## ground truth
            rgb = cv2.imread(rgb_path, -1)[:, :, ::-1]  ##bgr to rgb

            image_resized = resize_by_scaled_short_side(rgb, base_size, 1)
            h, w, _ = rgb.shape
            new_definitions = Background_Def  ## it is a dictionery contains iformation about labels and obtained from 'https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_2013.pdf'
            user_label = ['background', 'facade', 'sill', 'balcony', 'door', 'blind', 'molding', 'deco', 'shop',
                          'window',
                          'cornice', 'pillar']
            gt_embs_list = create_embs_from_names(user_label, new_definitions).float()
            prediction, logit_for_loss = single_scale_single_crop_cuda_train(model, image_resized, h, w,
                                                                             gt_embs_list=gt_embs_list, args=None)
            # prediction = torch.tensor(torch.from_numpy(np.transpose(prediction, (2, 0, 1))[None, ...]), device=device,
            #                           requires_grad=True)
            ground_truth = torch.from_numpy(anot_img_modif).cuda().long()
            loss = criterion(logit_for_loss.contiguous(), ground_truth.unsqueeze(0))
            loss_value += loss.item()
            loss.backward()
            tp, fp, fn, tn = smp.metrics.get_stats(logit_for_loss, ground_truth.unsqueeze(0),
                                                   mode='multilabel',
                                                   threshold=0.5)
            iou_value += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score_value += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            # use gradient accumulation for memory efficiency instead of real batch_size use synthetic batch_size
            if (i + 1) % batch_size == 0 or i + 1 == len(imgs_train_list):
                optimizer.step()
                optimizer.zero_grad()
                all_loss.append(loss_value / batch_size)
                losses.append(loss_value / batch_size)
                print(f'average loss per batch_size is(train) {loss_value / batch_size}')
                print(f'average iou per batch_size is(train) {iou_value / batch_size}')
                print(f'average f1_score per batch_size is(train) {f1_score_value / batch_size}')
                loss_value = 0
                iou_value = 0
                f1_score_value = 0
            ## evaluation of the model on test data each 100 iteration
            iou_test_value = 0
            f1_score_test_value = 0
            if (i + 1) % 100 == 0 or i + 1 == len(imgs_train_list):
                model.eval()
                for i, (rgb_test_path, anot_test_path) in enumerate(zip(imgs_test_list, anots_test_list)):
                    anot_img = cv2.imread(anot_test_path, cv2.IMREAD_COLOR)
                    ## convert anotation to acceptable format
                    anot_img_modif = anot_preproces(anot_img)  ## ground truth
                    rgb = cv2.imread(rgb_test_path, -1)[:, :, ::-1]  ##bgr to rgb
                    image_resized = resize_by_scaled_short_side(rgb, base_size, 1)
                    h, w, _ = rgb.shape
                    prediction = single_scale_single_crop_cuda(model, image_resized, h, w,
                                                               gt_embs_list=gt_embs_list, args=None)
                    ground_truth = torch.from_numpy(anot_img_modif[None, ...]).cuda().float()
                    prediction = torch.tensor(torch.from_numpy(np.transpose(prediction, (2, 0, 1))[None, ...]),
                                              device=device,
                                              requires_grad=False)
                    tp, fp, fn, tn = smp.metrics.get_stats(prediction, ground_truth.round().long(), mode='multilabel',
                                                           threshold=0.5)
                    iou_test_value += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                    f1_score_test_value += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
                print(f'average iou  is(test) {iou_test_value.item() / len(imgs_test_list)}')
                print(f'average f1_score per  is(test) {f1_score_test_value.item() / len(imgs_test_list)}')
                f1_score_test.append(f1_score_test_value.item() / len(imgs_test_list))
                iou_score_test.append(iou_test_value.item() / len(imgs_test_list))
                if iou_test_value.item() / len(imgs_test_list) > best_iou_score:
                    best_iou_score = iou_test_value.item() / len(imgs_test_list)
                    torch.save(model.state_dict(), 'new_segmentation_model.pth')

        print(f'average epoch_loss is {sum(losses) / len(losses)}')
        epoch_loss.append(sum(losses) / len(losses))
        print('f1_score_test', f1_score_test)
        print('iou_score_test', iou_score_test)
    print('all_train_loss', all_loss)
    print('epoch_loss for train', epoch_loss)
    plt.plot(all_loss, '-b', label='loss')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title('train_loss')

    plt.show()


# prediction = prediction.argmax(axis=-1).squeeze()
# probs = prediction.max(axis=-1).squeeze()
# high_prob_mask = probs > 0.5
#
# mask = high_prob_mask
# prediction[~mask] = 255
#
# pred_color = color_seg(prediction)
# vis_seg = visual_segments(pred_color, rgb)
#
# vis_seg.save(os.path.splitext(save_path)[0] + '_vis.png')
# cv2.imwrite(save_path, prediction.astype(np.uint8))


if __name__ == '__main__':
    train()

# args = get_parser()
# logger.info(args)
#
# dist_url = 'tcp://127.0.0.1:6769'
# dist_url = dist_url[:-2] + str(os.getpid() % 100).zfill(2)
# args.dist_url = dist_url
#
# num_gpus = torch.cuda.device_count()
# if num_gpus != args.gpus_num:
#     raise RuntimeError(
#         'The set gpus number cannot match the detected gpus number. Please check or set CUDA_VISIBLE_DEVICES')
#
# if num_gpus > 1:
#     args.distributed = True
# else:
#     args.distributed = False
#
# save_path = os.path.join(args.root_dir, args.save_folder, 'id2labels.json')
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# with open(save_path, 'w') as f:
#     json.dump(UNI_UID2UNAME, f)
#
# if not args.distributed:
#     main_worker(0, args)
# else:
#     mp.spawn(main_worker, nprocs=args.gpus_num, args=(args,))
