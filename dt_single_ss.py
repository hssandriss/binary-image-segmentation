import os
import random
import numpy as np
import argparse
import torch
import time
from torchvision.transforms import ToTensor
from utils.meters import AverageValueMeter
from utils.weights import load_from_weights
from utils import check_dir, set_random_seed, accuracy, mIoU, get_logger, save_in_log
from models.att_segmentation import AttSegmentator
from torch.utils.tensorboard import SummaryWriter
from data.transforms import get_transforms_binary_segmentation
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderSingleClassSemanticSegmentationVector, DataReaderSemanticSegmentationVector
import matplotlib.pyplot as plt
set_random_seed(0)
global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data")
    parser.add_argument('--pretrained_model_path', type=str, default='')
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--att', type=str, default='sdotprod',
                        help='Type of attention. Choose from {additive, cosine, dotprod, sdotprod}')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--snapshot-freq', type=int, default=5, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "att"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'dt_attseg', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)
    img_size = (args.size, args.size)

    # model
    pretrained_model = ResNet18Backbone(False)
    # TODO: Complete the documentation for AttSegmentator model
    # TODO: Build model AttSegmentator model
    model = AttSegmentator(2, pretrained_model.features, att_type='dotprod')

    if os.path.isfile(args.pretrained_model_path):
        model = load_from_weights(model, args.pretrained_model_path, logger)

    # dataset
    data_root = args.data_folder
    train_transform, val_transform, train_transform_mask, val_transform_mask = get_transforms_binary_segmentation(args)
    vec_transform = ToTensor()
    train_data = DataReaderSingleClassSemanticSegmentationVector(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_transform,
        vec_transform=vec_transform,
        target_transform=train_transform_mask
    )
    # Note that the dataloaders are different.
    # During validation we want to pass all the semantic classes for each image
    # to evaluate the performance.
    val_data = DataReaderSemanticSegmentationVector(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_transform,
        vec_transform=vec_transform,
        target_transform=val_transform_mask
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
                                               num_workers=6, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                             num_workers=6, pin_memory=True, drop_last=False)

    # TODO: loss
    criterion = torch.nn.CrossEntropyLoss()
    # TODO: SGD optimizer (see pretraining)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # TODO: loss function and SGD optimizer")

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_val_miou = 0.0

    train_losses = []
    val_losses = []
    val_iou = []

    for epoch in range(100):
        logger.info("Epoch {}".format(epoch))
        train_loss = train(train_loader, model, criterion, optimizer, logger)
        val_loss, val_miou = validate(val_loader, model, criterion, logger, epoch)

        # TODO save model
        logger.info("----------------------------------------------------------")
        logger.info("Epoch %d  train_loss %.3f val_loss: %.3f val_miou: %.3f" %
                    (epoch, train_loss, val_loss, val_miou))
        logger.info("----------------------------------------------------------")

        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            save_model(model, optimizer, args, epoch, val_loss, val_miou, logger, best=True)
        elif (val_miou > best_val_miou):
            best_val_miou = best_val_miou
            logger.info("saving weights...")
            save_model(model, optimizer, args, epoch, val_loss, val_miou, logger, best=True)
        else:
            logger.info("saving weights...")
            save_model(model, optimizer, args, epoch, val_loss, val_miou, logger, best=False)

        # Saving csv
        logger.info("saving results to csv...")
        np.savetxt('{}/train_seg_loss_{}.csv'.format(args.model_folder, args.exp_name),
                   np.array([train_losses]), delimiter=';')
        np.savetxt('{}/val_seg_loss_{}.csv'.format(args.model_folder, args.exp_name),
                   np.array([val_losses]), delimiter=';')
        np.savetxt('{}/val_seg_iou_{}.csv'.format(args.model_folder, args.exp_name),
                   np.array([val_iou]), delimiter=';')

    # Saving plots
    logger.info("saving results to png...")
    fig = plt.figure()
    plt.plot(np.arange(len(train_losses)), np.array([train_losses]), 'r', label="Training loss")
    plt.plot(np.arange(len(val_losses)), np.array([val_losses]), 'g', label="Validation loss")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.xlabel("average loss")
    plt.ylim(-1, 3)
    plt.title("Validation and training losses on the attention based segmentation")
    fig.savefig('{}/task_2_att_seg_{}.png'.format(args.model_folder, args.exp_name), dpi=300)
    # TODO: implement the code for saving the model"


def train(loader, model, criterion, optimizer, logger):
    logger.info("Training")
    model.train()
    batch_loss_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()
    time_meter = AverageValueMeter()
    steps_per_epoch = len(loader.dataset) / loader.batch_size

    start_time = time.time()
    batch_time = time.time()
    for idx, (img, v_class, label) in enumerate(loader):
        img = img.cuda()
        v_class = v_class.float().cuda().squeeze()
        logits, alphas = model(img, v_class, out_att=True)
        logits = logits.squeeze()
        labels = (torch.nn.functional.interpolate(label.cuda(), size=logits.shape[-2:]).squeeze(1)*256).long()
        loss = criterion(logits, labels)
        iou = mIoU(logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss_meter.add(loss.item())
        loss_meter.add(loss.item())
        iou_meter.add(iou)
        time_meter.add(time.time()-batch_time)

        if idx % 50 == 0 or idx == len(loader)-1:
            text_print = "Epoch {:.4f} Avg loss = {:.4f} mIoU = {:.4f} Time {:.2f} (Total:{:.2f}) Progress {}/{}".format(
                global_step / steps_per_epoch, loss_meter.mean, iou_meter.mean, time_meter.mean, time.time()-start_time, idx, int(steps_per_epoch))
            logger.info(text_print)
            loss_meter.reset()
            iou_meter.reset()

        batch_time = time.time()
    time_txt = "batch time: {:.2f} total time: {:.2f}".format(time_meter.mean, time.time()-start_time)
    logger.info(time_txt)
    return batch_loss_meter.mean


def validate(loader, model, criterion, logger, epoch=0):
    logger.info("Validating Epoch {}".format(epoch))
    model.eval()

    loss_meter = AverageValueMeter()
    iou_meter = AverageValueMeter()

    start_time = time.time()
    for idx, (img, v_class, label) in enumerate(loader):
        img = img.squeeze(0).cuda()
        v_class = v_class.float().cuda().squeeze()
        logits, alphas = model(img, v_class, out_att=True)
        label = label.squeeze(0).unsqueeze(1)
        labels = (torch.nn.functional.interpolate(label.cuda(), size=logits.shape[-2:]).squeeze(1)*256).long()
        loss = criterion(logits, labels)
        iou = mIoU(logits, labels)

        loss_meter.add(loss.item())
        iou_meter.add(iou)

    text_print = "Epoch {} Avg loss = {:.4f} mIoU = {:.4f} Time {:.2f}".format(
        epoch, loss_meter.mean, iou_meter.mean, time.time()-start_time)
    logger.info(text_print)
    return loss_meter.mean, iou_meter.mean


def save_model(model, optimizer, args, epoch, val_loss, val_iou, logger, best=False):
    # save model
    add_text_best = 'BEST' if best else ''
    logger.info('==> Saving '+add_text_best+' ... epoch %i loss %.3f miou %.3f ' % (epoch, val_loss, val_iou))
    state = {
        'opt': args,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': val_loss,
        'miou': val_iou
    }
    if best:
        torch.save(state, os.path.join(args.model_folder,
                                       'ckpt_epoch%i_loss%.3f_miou%.3f_best.pth' % (epoch, val_loss, val_iou)))
    else:
        torch.save(state, os.path.join(args.model_folder, 'ckpt_epoch%i_loss%.3f_miou%.3f.pth' % (epoch, val_loss, val_iou)))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
