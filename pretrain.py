import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from pprint import pprint
from data.pretraining import DataReaderPlainImg, custom_collate
from data.transforms import get_transforms_pretraining
from utils import check_dir, accuracy, get_logger
from models.pretraining_backbone import ResNet18Backbone
import matplotlib.pyplot as plt

# Run using Eg.
# py pretrain.py ./unlabelled_dataset/crops/images / --weights-init pretrain_weihts_init.pth --bs 25
global_step = 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--weights-init', type=str,
                        default="random")
    parser.add_argument('--output-root', type=str, default='./results')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--bs', type=int, default=8, help='batch_size')
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--snapshot-freq', type=int, default=1, help='how often to save models')
    parser.add_argument('--exp-suffix', type=str, default="", help="string to identify the experiment")
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "size"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(os.path.join(args.output_root, 'pretrain', args.exp_name))
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('this is my device: ', device)
    # build model and load weights
    model = ResNet18Backbone(pretrained=False).to(device)
    # raise NotImplementedError("TODO: load weight initialization")
    model.load_state_dict(torch.load(args.weights_init, map_location=device), strict=False)
    # load dataset
    data_root = args.data_folder
    train_transform, val_transform = get_transforms_pretraining(args)
    train_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "train"), transform=train_transform)
    val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.bs, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True, collate_fn=custom_collate)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True, collate_fn=custom_collate)

    # TODO: loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info('train_data {}'.format(train_data.__len__()))
    logger.info('val_data {}'.format(val_data.__len__()))

    best_val_loss = np.inf
    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    train_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(50):
        train_loss = train(train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)
        val_loss, val_acc = validate(val_loader, model, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print("Epoch %d  train_loss %.3f val_loss: %.3f val_acc: %.3f \n" % (epoch, train_loss, val_loss, val_acc))
        # save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "{}/task_1_{}_epoch_{}.pth".format(args.model_folder, args.exp_name, epoch))
    # Saving plots
    fig = plt.figure()
    plt.plot(np.arange(len(train_losses)), np.array([train_losses]).squeeze(), 'r', label="Training loss")
    plt.plot(np.arange(len(val_losses)), np.array([val_losses]).squeeze(), 'g', label="Validation loss")
    plt.legend(loc="upper right")
    plt.ylim(-1, 100)
    fig.savefig('{}/task_1_{}.png'.format(args.model_folder, args.exp_name), dpi=300)

# train one epoch over the whole training dataset. You can change the method's signature.


def train(loader, model, criterion, optimizer):
    epoch_loss = 0
    running_loss = 0
    count = 0
    for i, data in enumerate(loader, 0):
        img, label = data
        img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        epoch_loss += loss
        count += label.size(0)
        running_loss += loss
        loss.backward()
        optimizer.step()
        if not (i % 1000):
            print("training after %i batches with loss %.3f\n" % (i, running_loss / 1000))
            running_loss = 0
    return epoch_loss / count


# validation function. you can change the method's signature.
def validate(loader, model, criterion):
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).item()
    return loss/total, (100 * correct / total)


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
