import math
import os, time
import shutil
import torch
import csv
import vis_utils
from metrics import Result, Result_intensity

# from visdom import Visdom
from tensorboardX import SummaryWriter
import numpy as np
import scipy.ndimage.interpolation as itpl
import skimage.transform

fieldnames = ['epoch','rmse', 'photo', 'mae', 'irmse', 'imae',
                'mse', 'absrel', 'lg10',
                'silog','squared_rel',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']

class logger:
    def __init__(self, args, prepare = True):
        self.args = args
        output_directory = get_folder_name(args)
        self.output_directory = output_directory
        self.best_result = Result()
        self.best_result.set_to_worst()

        self.best_result_intensity = Result_intensity()
        self.best_result_intensity.set_to_worst()

        #visual
        # self.viz = Visdom(server='http://10.5.40.31', port=11207)
        # assert self.viz.check_connection()
        from datetime import datetime, timedelta, timezone
        utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
        cn_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        name_prefix = "log_lr{}bt{}decay{}wi{}wpure{}lradj{}/".format(args.lr ,args.batch_size, args.weight_decay, args.wi, args.wpure, args.lradj)
        #name_prefix = "log_lr{}bt{}decay{}_ws_{}_alphabeta_{}_{}/".format(args.lr ,args.batch_size, args.weight_decay, args.ws, args.alpha, args.beta)
        print(name_prefix + TIMESTAMP)
        self.writer = SummaryWriter(name_prefix + TIMESTAMP)

        if not prepare:
            return
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        self.train_csv = os.path.join(output_directory, 'train.csv')
        self.val_csv = os.path.join(output_directory, 'val.csv')
        self.best_txt = os.path.join(output_directory, 'best.txt')
        self.train_csv_intensity = os.path.join(output_directory, 'train_intensity.csv')
        self.val_csv_intensity = os.path.join(output_directory, 'val_intensity.csv')
  
        # backup the source code
        if args.resume == '':
            print("=> creating source code backup ...")
            backup_directory = os.path.join(output_directory, "code_backup")
            self.backup_directory = backup_directory
            backup_source_code(backup_directory)
            # create new csv files with only header
            with open(self.train_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(self.val_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(self.train_csv_intensity, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            with open(self.val_csv_intensity, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print("=> finished creating source code backup.")

    def conditional_print(self,split,i,epoch,lr,n_set,blk_avg_meter,avg_meter, typeDI="depth"):
        if (i + 1) % self.args.print_freq == 0:
            avg = avg_meter.average()
            blk_avg = blk_avg_meter.average()
            print('=> output: {}'.format(self.output_directory))
            print('{typeDI} {split} Epoch: {0} [{1}/{2}]\tlr={lr} '
                  't_Data={blk_avg.data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={blk_avg.gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={blk_avg.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={blk_avg.mae:.2f}({average.mae:.2f}) '
                  'iRMSE={blk_avg.irmse:.2f}({average.irmse:.2f}) '
                  'iMAE={blk_avg.imae:.2f}({average.imae:.2f})\n\t'
                  'silog={blk_avg.silog:.2f}({average.silog:.2f}) '
                  'squared_rel={blk_avg.squared_rel:.2f}({average.squared_rel:.2f}) '
                  'Delta1={blk_avg.delta1:.3f}({average.delta1:.3f}) '
                  'REL={blk_avg.absrel:.3f}({average.absrel:.3f})\n\t'
                  'Lg10={blk_avg.lg10:.3f}({average.lg10:.3f}) '
                  'Photometric={blk_avg.photometric:.3f}({average.photometric:.3f}) '.format(
                  epoch, i+1, n_set, lr=lr, blk_avg=blk_avg, average=avg,
                  split=split.capitalize(), typeDI = typeDI)
            )
            blk_avg_meter.reset()

    def conditional_save_info(self, split, average_meter, epoch):
        avg = average_meter.average()
        if split == "train":
            csvfile_name = self.train_csv
        elif split == "val":
            csvfile_name = self.val_csv
        elif split == "eval":
            eval_filename = os.path.join(self.output_directory, 'eval.txt')
            self.save_single_txt(eval_filename, avg, epoch)
            return avg
        elif "test" in split:
            return avg
        else:
            raise ValueError("wrong split provided to logger")
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow( {'epoch': epoch, 'rmse': avg.rmse,
            				  'photo': avg.photometric,
                              'mae': avg.mae, 'irmse': avg.irmse,
                              'imae': avg.imae, 'mse': avg.mse,
                              'silog': avg.silog,
                              'squared_rel': avg.squared_rel,
                              'absrel': avg.absrel, 'lg10': avg.lg10,
                              'delta1': avg.delta1, 'delta2': avg.delta2,
                              'delta3': avg.delta3, 'gpu_time': avg.gpu_time,
                              'data_time': avg.data_time})
        return avg

    def conditional_save_info_intensity(self, split, average_meter, epoch):
        avg = average_meter.average()
        if split == "train":
            csvfile_name = self.train_csv_intensity
        elif split == "val":
            csvfile_name = self.val_csv_intensity
        elif split == "eval":
            eval_filename = os.path.join(self.output_directory, 'eval.txt')
            self.save_single_txt(eval_filename, avg, epoch)
            return avg
        elif "test" in split:
            return avg
        else:
            raise ValueError("wrong split provided to logger")
        with open(csvfile_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow( {'epoch': epoch, 'rmse': avg.rmse,
            				  'photo': avg.photometric,
                              'mae': avg.mae, 'irmse': avg.irmse,
                              'imae': avg.imae, 'mse': avg.mse,
                              'silog': avg.silog,
                              'squared_rel': avg.squared_rel,
                              'absrel': avg.absrel, 'lg10': avg.lg10,
                              'delta1': avg.delta1, 'delta2': avg.delta2,
                              'delta3': avg.delta3, 'gpu_time': avg.gpu_time,
                              'data_time': avg.data_time})
        return avg

    def save_single_txt(self, filename, result, epoch):
        with open(filename, 'w') as txtfile:
            txtfile.write(("rank_metric={}\n" +
                            "epoch={}\n" +
                            "rmse={:.3f}\n" +
                            "mae={:.3f}\n" +
                            "silog={:.3f}\n" +
                            "squared_rel={:.3f}\n" +
                            "irmse={:.3f}\n" +
                            "imae={:.3f}\n" +
                            "mse={:.3f}\n" +
                            "absrel={:.3f}\n" +
                            "lg10={:.3f}\n" +
                            "delta1={:.3f}\n" +
                            "t_gpu={:.4f}").format(
                          self.args.rank_metric, epoch,result.rmse, result.mae,
                          result.silog, result.squared_rel,
                          result.irmse, result.imae, result.mse, result.absrel,
                          result.lg10, result.delta1, result.gpu_time))
    def save_single_txt_with_intensity(self, filename, result, result_intensity, epoch):
        with open(filename, 'w') as txtfile:
            txtfile.write(("rank_metric={}\n" +
                            "epoch={}\n" +
                            "depth_rmse={:.3f}\n" +
                            "mae={:.3f}\n" +
                            "silog={:.3f}\n" +
                            "squared_rel={:.3f}\n" +
                            "irmse={:.3f}\n" +
                            "imae={:.3f}\n" +
                            "mse={:.3f}\n" +
                            "absrel={:.3f}\n" +
                            "lg10={:.3f}\n" +
                            "delta1={:.3f}\n" +
                            "t_gpu={:.4f}\n" + "wi={}\n" + 
                            "btsize={}\n" + "intensity_rmse={:.3f}\n" +
                            "intensity_irmse={:.3f}\n" + "intensity_mae={:.3f}\n" + "intensity_imae={:.3f}\n").format(
                          self.args.rank_metric, epoch,result.rmse, result.mae,
                          result.silog, result.squared_rel,
                          result.irmse, result.imae, result.mse, result.absrel,
                          result.lg10, result.delta1, result.gpu_time, self.args.wi,
                          self.args.batch_size, result_intensity.rmse,
                          result_intensity.irmse, result_intensity.mae, result_intensity.imae))

    def save_best_txt(self, result, epoch):
        self.save_single_txt(self.best_txt, result, epoch)

    def save_best_txt_with_intensity(self, result, result_intensity, epoch):
        self.save_single_txt_with_intensity(self.best_txt, result,result_intensity, epoch)

    def _get_img_comparison_name(self, mode, epoch, is_best = False):
        if mode == 'eval':
            return self.output_directory + '/comparison_eval.png'
        if mode == 'val':
          if is_best:
              return self.output_directory + '/comparison_best.png'
          else:
              return self.output_directory + '/comparison_' + str(epoch) + '.png'

    # ele-batch_data
    def conditional_save_img_comparison(self, mode, i, ele, pred, epoch):
        # save 8 images for visualization
        if mode == 'val' or mode == 'eval':
            skip = 100
            if i == 0:
                self.img_merge = vis_utils.merge_into_row(ele, pred)
            elif i % skip ==0 and i < 8*skip:
                row = vis_utils.merge_into_row(ele, pred)
                self.img_merge = vis_utils.add_row(self.img_merge, row)
            elif i == 8*skip:
                filename = self._get_img_comparison_name(mode, epoch)
                vis_utils.save_image(self.img_merge, filename)  #HWC
                # input C x H x W
                img_np_rescale = skimage.transform.rescale(np.array(self.img_merge, dtype = 'float64'), 0.5, order=0)
                img_np_rescale_CHW = np.transpose(img_np_rescale, (2, 0,1))
                self.writer.add_image('eval_comparison', img_np_rescale_CHW, i)

    def conditional_save_img_comparison_with_intensity(self, mode, i, ele, pred, pred_intensity, epoch):
        # save 8 images for visualization
        if mode == 'val' or mode == 'eval':
            skip = 100
            if i == 0:
                self.img_merge = vis_utils.merge_into_row_with_intensity(ele, pred, pred_intensity)
                # self.img_merge = vis_utils.merge_into_row(ele, pred)
            elif i % skip ==0 and i < 8*skip:
                # row = vis_utils.merge_into_row(ele, pred)
                row = vis_utils.merge_into_row_with_intensity(ele, pred, pred_intensity)
                self.img_merge = vis_utils.add_row(self.img_merge, row)
            elif i == 8*skip:
                filename = self._get_img_comparison_name(mode, epoch)
                vis_utils.save_image(self.img_merge, filename)  #HWC
                # input C x H x W
                img_np_rescale = skimage.transform.rescale(np.array(self.img_merge, dtype = 'float64'), 0.5, order=0)
                img_np_rescale_CHW = np.transpose(img_np_rescale, (2, 0,1))
                self.writer.add_image('comparison', img_np_rescale_CHW, i)

    def conditional_save_img_comparison_with_intensity2(self, mode, i, ele, pred, pred_pure, pred_intensity, epoch):
        # save 8 images for visualization
        if mode == 'val' or mode == 'eval':
            skip = 100
            if i == 0:
                self.img_merge = vis_utils.merge_into_row_with_intensity2(ele, pred, pred_pure, pred_intensity)
                # self.img_merge = vis_utils.merge_into_row(ele, pred)
            elif i % skip ==0 and i < 8*skip:
                # row = vis_utils.merge_into_row(ele, pred)
                row = vis_utils.merge_into_row_with_intensity2(ele, pred, pred_pure, pred_intensity)
                self.img_merge = vis_utils.add_row(self.img_merge, row)
            elif i == 8*skip:
                filename = self._get_img_comparison_name(mode, epoch)
                vis_utils.save_image(self.img_merge, filename)  #HWC
                # input C x H x W
                img_np_rescale = skimage.transform.rescale(np.array(self.img_merge, dtype = 'float64'), 0.5, order=0)
                img_np_rescale_CHW = np.transpose(img_np_rescale, (2, 0,1))
                self.writer.add_image('comparison', img_np_rescale_CHW, i)

    def save_img_comparison_as_best(self, mode, epoch):
        if mode == 'val':
            filename = self._get_img_comparison_name(mode, epoch, is_best = True)
            vis_utils.save_image(self.img_merge, filename)
            self.writer.add_image('val_comparison_best', np.transpose(self.img_merge, (2, 0, 1)))
                
    def get_ranking_error(self,result):
        return getattr(result,self.args.rank_metric)

    def rank_conditional_save_best(self, mode, result, epoch):
        error = self.get_ranking_error(result)
        best_error = self.get_ranking_error(self.best_result)
        is_best = error < best_error
        if is_best and mode == "val":
            self.old_best_result = self.best_result
            self.best_result = result
            self.save_best_txt(result, epoch)
        return is_best

    def rank_conditional_save_best_with_intensity(self, mode, result,result_intensity, epoch):
        error_depth = self.get_ranking_error(result)
        error_intensity = self.get_ranking_error(result_intensity)
        error = error_depth / 1000.0 + error_intensity / 5.0
        best_error_depth = self.get_ranking_error(self.best_result)
        best_error_intensity = self.get_ranking_error(self.best_result_intensity)
        best_error = best_error_depth / 1000.0 + best_error_intensity / 5.0
        is_best = error < best_error
        if is_best and mode == "val":
            self.old_best_result = self.best_result
            self.old_best_result_intensity = self.best_result_intensity
            self.best_result = result
            self.best_result_intensity = result_intensity
            self.save_best_txt_with_intensity(result, result_intensity, epoch)
            self.writer.add_scalar("eval/best_rmseD", result.rmse, epoch)
            self.writer.add_scalar("eval/best_rmseI", result_intensity.rmse, epoch)
            self.writer.add_scalar("eval/best_rmseTotal", result.rmse / 1000.0 + result_intensity.rmse / 5.0, epoch)
        return is_best
        
    def conditional_save_pred_named_with_intensity(self, mode, name, pred, pred_intensity, epoch):
        # if ("test" in mode or mode == "eval") and self.args.save_pred:
        if ("test" in mode or mode == "val") or mode=='eval':

            # save images for visualization/ testing
            image_folder = os.path.join(self.output_directory, mode+"_output_depth")
            # print(name, image_folder, pred.shape)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred.data.cpu()).numpy()
            # filename = os.path.join(image_folder, '{0:010d}.png'.format(i))
            filename = os.path.join(image_folder, name)
            vis_utils.save_depth_as_uint16png(img, filename)

            image_folder = os.path.join(self.output_directory, mode+"_output_intensity")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred_intensity.data.cpu()).numpy()
            # filename = os.path.join(image_folder, '{0:010d}.png'.format(i))
            filename = os.path.join(image_folder, name)
            vis_utils.save_depth_as_uint16png(img, filename)

    def conditional_save_pred_named(self, mode, name, pred, epoch):
        # if ("test" in mode or mode == "eval") and self.args.save_pred:
        if ("test" in mode or mode == "val") or mode=='eval':

            # save images for visualization/ testing
            image_folder = os.path.join(self.output_directory, mode+"_output")
            # print(name, image_folder)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred.data.cpu()).numpy()
            # filename = os.path.join(image_folder, '{0:010d}.png'.format(i))
            filename = os.path.join(image_folder, name)
            vis_utils.save_depth_as_uint16png(img, filename)

    def conditional_save_pred(self, mode, i, pred, epoch):
        if ("test" in mode or mode == "eval") and self.args.save_pred:

            # save images for visualization/ testing
            image_folder = os.path.join(self.output_directory, mode+"_output")
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            img = torch.squeeze(pred.data.cpu()).numpy()
            filename = os.path.join(image_folder, '{0:010d}.png'.format(i))
            vis_utils.save_depth_as_uint16png(img, filename)

    def conditional_summarize(self, mode, avg, is_best):
        print("\n*\nSummary of ", mode, "round")
        print(''
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Photo={average.photometric:.3f}\n'
        'iRMSE={average.irmse:.3f}\n'
        'iMAE={average.imae:.3f}\n'
        'squared_rel={average.squared_rel}\n'
        'silog={average.silog}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}'.format(
        average=avg, time=avg.gpu_time))
        if is_best and mode == "val":
            print("New best model by %s (was %.3f)"%
              (self.args.rank_metric,
               self.get_ranking_error(self.old_best_result)))
        elif mode == "val":
            print("(best %s is %.3f)"%
              (self.args.rank_metric,
              self.get_ranking_error(self.best_result)))
        print("*\n")

    def conditional_summarize_intensity(self, mode, avg):
        print("\n*\nIntensity Summary of ", mode, "round")
        print(''
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Photo={average.photometric:.3f}\n'
        'iRMSE={average.irmse:.3f}\n'
        'iMAE={average.imae:.3f}\n'
        'squared_rel={average.squared_rel}\n'
        'silog={average.silog}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}'.format(
        average=avg, time=avg.gpu_time))
        print("*\n")
            
ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*", "log*", "*.tar")
#ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*")
def backup_source_code(backup_directory):
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree('.', backup_directory, ignore=ignore_hidden)

def adjust_learning_rate(lr_init, optimizer, epoch, adj):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    # lr = lr_init * (0.1 ** (epoch // 5))
    if adj == 0:
        return 0
    lr = lr_init * (0.1 ** (epoch // adj))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def get_folder_name(args):
    current_time = time.strftime('%Y-%m-%d@%H-%M')
    if args.use_pose:
        prefix = "mode={}.w1={}.w2={}.".format(args.train_mode, args.w1, args.w2)
    else:
        prefix = "mode={}.wi={}.wpure={}.".format(args.train_mode, args.wi, args.wpure)
    return os.path.join(args.result,
        prefix + 'input={}.resnet{}.criterion={}.lr={}adj={}.bs={}.wd={}.pretrained={}.jitter={}.time={}'.
        format(args.input, args.layers, args.criterion, \
            args.lr, args.lradj, args.batch_size, args.weight_decay, \
            args.pretrained, args.jitter, current_time
            ))

avgpool = torch.nn.AvgPool2d(kernel_size=2, stride=2).cuda()
def multiscale(img):
    img1 = avgpool(img)
    img2 = avgpool(img1)
    img3 = avgpool(img2)
    img4 = avgpool(img3)
    img5 = avgpool(img4)
    return img5, img4, img3, img2, img1
