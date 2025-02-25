import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
import torch.utils.data

from dataloaders.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
from model import DepthCompletionNet
from metrics import AverageMeter, AverageIntensity, Result, Result_intensity
import criteria
import helper
from inverse_warp import Intrinsics, homography_from

import numpy as np

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=11, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) + ' (default: l2)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-i','--input', type=str, default='gd',
                    choices=input_options, help='input: | '.join(input_options))
parser.add_argument('-l','--layers', type=int, default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('--pretrained', action="store_true",
                    help='use ImageNet pre-trained weights')
parser.add_argument('--val', type=str, default="select",
                    choices= ["select","full"], help='full or select validation set')
parser.add_argument('--jitter', type=float, default=0.1,
                    help = 'color jitter for images')
parser.add_argument('--rank-metric', type=str, default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help = 'metrics for which best result is sbatch_datacted')
parser.add_argument('-m', '--train-mode', type=str, default="dense",
                    choices = ["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
                    help = 'dense | sparse | photo | sparse+photo | dense+photo')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('--wi', default=0.02, type=float)
parser.add_argument('--wpure', default=0.02, type=float)
parser.add_argument('--lradj', default=0, type=float)
parser.add_argument('--DI2DI', default='', type=str, metavar='PATH',
                    help='path to pretrian DI2DI (default: none)')

args = parser.parse_args()
args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input) or args.use_pose
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0
# args.wi = 0.02
print(args)

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (args.criterion == 'l2') else criteria.MaskedL1Loss()
intensity_criterion = criteria.MaskedMSELoss() if (args.criterion == 'l2') else criteria.MaskedL1Loss()
intensity_pure_criterion = criteria.MaskedMSELoss() if (args.criterion == 'l2') else criteria.MaskedL1Loss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()

if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib()
    fu, fv = float(K[0,0]), float(K[1,1])
    cu, cv = float(K[0,2]), float(K[1,2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv).cuda()

batch_num = 0
def iterate(mode, args, loader, model, optimizer, logger, epoch):
    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    block_average_meter_intensity = AverageIntensity()
    average_meter_intensity = AverageIntensity()
    meters_intensity = [block_average_meter_intensity, average_meter_intensity]

    block_average_meter_intensity_pure = AverageIntensity()
    average_meter_intensity_pure = AverageIntensity()
    meters_intensity_pure = [block_average_meter_intensity_pure, average_meter_intensity_pure]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        # lr = args.lr #helper.adjust_learning_rate(args.lr, optimizer, epoch)
        if args.lradj > 0:
            lr = helper.adjust_learning_rate(args.lr, optimizer, epoch, args.lradj)
        else:
            lr = args.lr 
    else:
        model.eval()
        lr = 0

    global batch_num
    for i, batch_data in enumerate(loader):
        if mode == "train":
            batch_num += 1
        start = time.time()
        file_name = batch_data['filename']

        batch_data = {key:val.cuda() for key,val in batch_data.items() if (key != "filename" and val is not None)}
    #    batch_data = {key:val.cuda() for key,val in batch_data.items() if val is not None}
        gt = batch_data['gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        #   Ireal as gt_intensity
        gt_intensity = batch_data['gt_intensity'] if mode != 'test_prediction' and mode != 'test_completion' else None
        gt_intensity_pure = batch_data['gt_intensity_pure'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - start

        start = time.time()
        pred = model(batch_data)

        pred_intensity_pure = pred[0][:,1,:,:].unsqueeze(1)
        pred_intensity = pred[1]
        pred = pred[0][:,0,:,:].unsqueeze(1)

        depth_loss, intensity_loss, photometric_loss, smooth_loss, pure_loss, mask = 0, 0, 0, 0, 0, None
        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                pure_loss = intensity_pure_criterion(pred_intensity_pure, gt_intensity_pure)
                intensity_loss = intensity_criterion(pred_intensity, gt_intensity)
                mask = (gt < 1e-3).float()

            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = kitti_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    warped_ = homography_from(rgb_near_, pred_, batch_data['r_mat'], batch_data['t_vec'], intrinsics_)
                    photometric_loss += photometric_criterion(rgb_curr_, warped_, mask_) * (2**(scale-num_scales))

            # Loss 3: the depth smoothness loss
            smooth_loss = smoothness_criterion(pred) if args.w2>0 else 0

            # backprop
            loss = depth_loss + args.wi * intensity_loss + args.wpure * pure_loss
            # loss = depth_loss + wi * intensity_loss + args.w1*photometric_loss + args.w2*smooth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #log train
            if batch_num % 25 == 0 and mode == "train":
                logger.writer.add_scalar('train/loss_total', loss, batch_num)
                logger.writer.add_scalar('train/loss_depth', depth_loss, batch_num)
                logger.writer.add_scalar('train/loss_intensity', intensity_loss, batch_num)
                logger.writer.add_scalar('train/loss_pure', pure_loss, batch_num)
                logger.writer.add_scalar('train/lr', lr, batch_num)
                #if batch_num % 200 == 0:
                #    logger.writer.add_image('train/Ipure', pred_intensity_pure[0], batch_num)
            
        gpu_time = time.time() - start

        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            # result_intensity = Result()
            result_intensity_pure = Result_intensity()
            result_intensity = Result_intensity()
            if mode != 'test_prediction' and mode != 'test_completion':
                # result.evaluate(pred.data, gt.data, photometric_loss)
                result.evaluate(pred.data, gt.data, photometric_loss)
                result_intensity_pure.evaluate(pred_intensity_pure.data, gt_intensity_pure.data)
                result_intensity.evaluate(pred_intensity.data, gt_intensity.data)

            [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
            [m_intensity.update(result_intensity, gpu_time, data_time, mini_batch_size) for m_intensity in meters_intensity]
            [m_intensity_pure.update(result_intensity_pure, gpu_time, data_time, mini_batch_size) for m_intensity_pure in meters_intensity_pure]
            
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            # logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
            logger.conditional_save_pred_named(mode, file_name[0], pred, epoch)

            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter_intensity_pure, average_meter_intensity_pure, "Ipure")
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter_intensity, average_meter_intensity, "Intensity")
            # logger.conditional_save_img_comparison_with_intensity(mode, i, batch_data, pred, pred_intensity, epoch)
            logger.conditional_save_img_comparison_with_intensity2(mode, i, batch_data, pred, pred_intensity_pure, pred_intensity, epoch)
            # run when eval, bt always = 1
            #logger.conditional_save_pred_named_with_intensity(mode, file_name[0], pred, pred_intensity, epoch)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    avg_intensity = logger.conditional_save_info_intensity(mode, average_meter_intensity, epoch)
    # is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    is_best = logger.rank_conditional_save_best_with_intensity(mode, avg, avg_intensity, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    logger.conditional_summarize_intensity(mode, avg_intensity)

    return avg, avg_intensity, is_best

def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            args = checkpoint['args']
            is_eval = True
            print("=> checkpoint loaded.")
        else:
            print("=> no model found at '{}'".format(args.evaluate))
            return
    elif args.resume: # optionally resume from a checkpoint
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer...")
    model = DepthCompletionNet(args).cuda()
    model_named_params = [p for _,p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)
    print("=> model and optimizer created.")
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    model = torch.nn.DataParallel(model)
    print("=> model transferred to multi-GPU.")

    # Data loading code
    print("=> creating data loaders ...")
    if not is_eval:
        train_dataset = KittiDepth('train', args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)
    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=2, pin_memory=False) # set batch size to be 1 for validation
    print("=> data loaders created.")

    # create backups and results folde
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    if is_eval:
        result, result_intensity, is_best = iterate("val", args, val_loader, model, None, logger, checkpoint['epoch'])
        return

    # main loop

    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch) # train for one epoch
        result, result_intensity, is_best = iterate("val", args, val_loader, model, None, logger, epoch) # evaluate on validation set
        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)

        logger.writer.add_scalar('eval/rmse_depth', result.rmse, epoch)
        logger.writer.add_scalar('eval/rmse_intensity', result_intensity.rmse, epoch)
        logger.writer.add_scalar('eval/mae_depth', result.mae, epoch)
        logger.writer.add_scalar('eval/mae_intensity', result_intensity.mae, epoch)
       # logger.writer.add_scalar('eval/irmse_depth', result.irmse, epoch)
       # logger.writer.add_scalar('eval/irmse_intensity', result_intensity.irmse, epoch)
        logger.writer.add_scalar('eval/rmse_total', result.rmse + args.wi * result_intensity.rmse, epoch)


if __name__ == '__main__':
    main()

