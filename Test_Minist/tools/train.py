from typing import Dict, List

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from segmentation_models_pytorch.losses.jaccard import JaccardLoss
from Test_Minist.utils.transforms_utils import get_imagenet_mean_std, normalize_img, pad_to_crop_sz, \
    resize_by_scaled_short_side


class CMPFacade(Dataset):

    def __init__(self, data_dir: str = "/home/amirhossein/PycharmProjects/SSIW/CMP_facade_DB_base/base",
                 transform: List = None, resize_base_size=512):
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



def train(data_dir: str = '/home/amirhossein/PycharmProjects/SSIW/CMP_facade_DB_base/base'):
    imgs_list = sorted(glob.glob(data_dir + "/*.jpg"))
    anots_list = sorted(glob.glob(data_dir + "/*.png"))
    for i, rgb_path in tqdm(enumerate(zip(imgs_list, anots_list))):
        EPOCH = 10
        for i in range(EPOCH):
            # save_path = os.path.join(args.root_dir, args.save_folder, os.path.basename(rgb_path))
            # save_path = os.path.splitext(save_path)[0] + '.png'

            # os.makedirs(os.path.dirname(save_path), exist_ok=True)

            rgb = cv2.imread(rgb_path, -1)[:, :, ::-1]
            image_resized = resize_by_scaled_short_side(rgb, args.base_size, 1)
            h, w, _ = rgb.shape

            if args.single_scale_single_crop:
                out_logit = single_scale_single_crop_cuda(model, image_resized, h, w, gt_embs_list=gt_embs_list,
                                                          args=args)
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


if __name__ == '__main__':
    img_height = 1024
    img_width = 543
    mean, std = get_imagenet_mean_std()
    transforms_ = [
        transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
        transforms.RandomCrop((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    dataset = CMPFacade(transform=transforms_)
    train_dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )
    print(dataset.img_files)
    print(dataset)
    for data in train_dataloader:
        print(data)
