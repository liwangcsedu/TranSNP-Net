import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.TranSNP_Net import TranSNP_Net
from data import test_dataset
from evaluation.eval import eval

def test_model(model_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=384, help='testing size')
    parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
    parser.add_argument('--test_path', type=str, default='./data/test_datasets/', help='test dataset path')
    opt = parser.parse_args()

    dataset_path = opt.test_path

    # set device for test
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')

    # load the model
    model = TranSNP_Net()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.cuda()
    model.eval()

    test_datasets = ['STERE-797', 'DUT-RGBD', 'SSD', 'ReDWeb', 'DES', 'LFSD', 'NJU2K', 'NLPR', 'SIP', 'STERE', 'RGBD135']

    for dataset in test_datasets:
        save_path = './test_maps/TranSNP_Net/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        depth_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth = depth.repeat(1, 3, 1, 1).cuda()
            res, res2, res3, res4 = model(image, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            print('save img to: ', save_path + name)
            cv2.imwrite(save_path + name, res * 255)
        print('Test Done!')


if __name__ == '__main__':
    #test
    test_model('./cpts/TranSNP_Net_epoch_best.pth')

    #eval
    eval('./cpts/TranSNP_Net_epoch_best.pth')
