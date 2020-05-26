import matplotlib.pyplot as plt
from data.segmentation import DataReaderBinarySegmentation
from models.pretraining_backbone import ResNet18Backbone
from data.transforms import get_transforms_binary_segmentation
from models.second_segmentation import Segmentator
from utils import check_dir, set_random_seed, accuracy, mIoU, get_logger
import torch
import argparse
import numpy as np
import os
from pprint import pprint
import random
import sys
import time
sys.path.insert(0, os.getcwd())
set_random_seed(0)
global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str,
                        help="folder containing the data")
    parser.add_argument('--weights-init', type=str, default="ImageNet")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--snapshot-freq', type=int,
                        default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="",
                        help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs"]
    args.exp_name = "_".join(
        ["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(
        args.output_root, 'dt_binseg', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)
    img_size = (args.size, args.size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('this is my device: ', device)
    # model
    pretrained_model = ResNet18Backbone(pretrained=False).to(device)
    pretrained_model.load_state_dict(torch.load(
        args.weights_init, map_location=device)['model'])
    model = Segmentator(2, pretrained_model.features, img_size).cuda()

    # dataset
    train_trans, val_trans, train_target_trans, val_target_trans = get_transforms_binary_segmentation(
        args)
    data_root = args.data_folder
    train_data = DataReaderBinarySegmentation(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_trans,
        target_transform=train_target_trans
    )
    val_data = DataReaderBinarySegmentation(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_trans,
        target_transform=val_target_trans
    )
    print("Dataset size: {} samples".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True,
                                               num_workers=6, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,
                                             num_workers=6, pin_memory=True, drop_last=False)

    # TODO: loss ()
    criterion = torch.nn.CrossEntropyLoss()
    # TODO: SGD optimizer (see pretraining)
    # optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # TODO: loss function and SGD optimizer"

    expdata = "  \n".join(["{} = {}".format(k, v)
                           for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    best_val_miou = 0.0

    # TODO save model
    train_losses = []
    train_iou = []
    val_losses = []
    val_iou = []
    for epoch in range(50):
        logger.info("Epoch {}".format(epoch))
        train_loss, train_miou = train(train_loader, model, criterion,
                                       optimizer, epoch, logger)
        train_losses.append(train_loss)
        train_iou.append(train_miou)

        val_loss, val_miou = validate(val_loader, model, criterion, logger, epoch)
        val_losses.append(val_loss)
        val_iou.append(val_miou)

        logger.info(
            "----------------------------------------------------------")
        logger.info("Epoch %d  train_loss %.3f train_miou: %.3f val_loss: %.3f val_miou: %.3f" %
                    (epoch, train_loss, train_miou, val_loss, val_miou))
        logger.info(
            "----------------------------------------------------------")

        if (val_loss < best_val_loss):
            best_val_loss = val_loss
            logger.info("Model with best validation loss found!")
            save_model(model, optimizer, args, epoch,
                       val_loss, val_miou, logger, best=True)
        elif (val_miou > best_val_miou):
            best_val_miou = val_miou
            logger.info("Model with best validation miou found!")
            save_model(model, optimizer, args, epoch,
                       val_loss, val_miou, logger, best=True)

        # Saving csv
        logger.info("saving results to csv...")

        np.savetxt('{}/train_bin_loss_{}.csv'.format(args.model_folder, args.exp_name),
                   np.array([train_losses]), delimiter=';')
        np.savetxt('{}/train_bin_iou_{}.csv'.format(args.model_folder, args.exp_name),
                   np.array([train_iou]), delimiter=';')
        np.savetxt('{}/val_bin_loss_{}.csv'.format(args.model_folder, args.exp_name),
                   np.array([val_losses]), delimiter=';')
        np.savetxt('{}/val_bin_iou_{}.csv'.format(args.model_folder, args.exp_name),
                   np.array([val_iou]), delimiter=';')

    # Saving plots
    # logger.info("saving results to png...")
    # fig = plt.figure()
    # plt.plot(np.arange(len(train_losses)), np.array(
    #     [train_losses]).squeeze(), 'r', label="Training loss")
    # plt.plot(np.arange(len(val_losses)), np.array(
    #     [val_losses]).squeeze(), 'g', label="Validation loss")
    # plt.legend(loc="upper right")
    # plt.xlabel("epoch")
    # plt.xlabel("average loss")
    # plt.ylim(-1, 3)
    # plt.title("Validation and training losses on the binary segmentation")
    # fig.savefig('{}/task_1_binseg_{}.png'.format(args.model_folder,
    #                                              args.exp_name), dpi=300)


def train(loader, model, criterion, optimizer, epoch, logger):
    # TODO: training routine
    model.train()
    epoch_loss = 0.
    running_loss = 0.
    epoch_miou = 0
    count = 0
    for i, data in enumerate(loader, 0):
        images, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()
        outputs = model(images)
        labels = (labels.squeeze() * 255).long()
        loss = criterion(outputs, labels).mean()
        epoch_loss += loss.item()
        running_loss += loss.item()
        count += 1
        epoch_miou += mIoU(outputs, labels).item()
        loss.backward()
        optimizer.step()
        if (i % 100 == 99):
            logger.info("Epoch %i training iter %i with loss %f" %
                        (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0
    return epoch_loss / count, 100 * epoch_miou / count


def validate(loader, model, criterion, logger, epoch=0):
    # TODO: validation routine
    model.eval()
    val_loss = 0.
    val_miou = 0.
    count = 0
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            image, label = data[0].cuda(), data[1].cuda()
            output = model(image)
            label = label * 255
            label = torch.nn.functional.interpolate(
                label, (output.shape[2], output.shape[3]))
            label = label.squeeze(1).long()
            val_loss += criterion(output, label).mean().item()
            val_miou += mIoU(output, label).item()
            count += 1
    return val_loss / count, (100 * val_miou / count)
    # return mean_val_loss, mean_val_iou


def save_model(model, optimizer, args, epoch, val_loss, val_iou, logger, best=False):
    # save model
    add_text_best = 'BEST' if best else ''
    logger.info('==> Saving '+add_text_best +
                ' ... epoch %i loss %.3f miou %.3f ' % (epoch, val_loss, val_iou))
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
        torch.save(state, os.path.join(args.model_folder,
                                       'ckpt_epoch%i_loss%.3f_miou%.3f.pth' % (epoch, val_loss, val_iou)))


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
