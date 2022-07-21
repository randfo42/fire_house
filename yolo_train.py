import math

import torch
import torch.nn as nn

from yolov3 import load_model
import argparse
import os
import torch
from data import *
from utils.augmentations import SSDAugmentation
import torch.backends.cudnn as cudnn
import torch.optim as optim
from utils.loss import compute_loss
import torch.utils.data
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
device = '0,1'
#torch.cuda.set_device('cuda:0,1')

parser = argparse.ArgumentParser(description="Trains the YOLO model.")
parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
# parser.add_argument('--gamma', default=0.1, type=float,
#                     help='Gamma update for SGD')

args = parser.parse_args()
# args = parser.parse_args([])


def train(args):
    yolo_net = load_model(args.model, args.pretrained_weights)
    
    print(COCO_ROOT)

    if os.path.exists(COCO_ROOT):
        args.dataset_root = COCO_ROOT
        cfg = coco512
        dataset = COCODetection(root=args.dataset_root,
                                    transform=SSDAugmentation(cfg['min_dim'],
                                                              MEANS))


    net = yolo_net
    # net = torch.nn.DataParallel(yolo_net, device_ids=[0,1])
    # cudnn.benchmark = True
    net = net.cuda()


    optimizer = optim.Adam(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    data_loader =torch.utils.data.DataLoader(dataset, 8,  # TODO? batch size 
                                  num_workers=2,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    for epoch in range(0,args.epochs):
        for data in tqdm(data_loader):

            images, targets = data
            images  = torch.tensor(images)


            images = images.cuda()
            out = net(images)

            for i,val in enumerate(targets):
                idx = torch.ones(targets[i].size()[0],1)*i

                targets[i] = torch.cat([idx,targets[i]],dim=1)
            targets = torch.cat(targets).cuda()
            loss, loss_components = compute_loss(out, targets,net)
            optimizer.zero_grad()
            loss.backward()

if __name__ == '__main__':
    train(args)

