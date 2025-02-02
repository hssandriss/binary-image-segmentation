import os
from shutil import copyfile
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from torchvision.utils import save_image
from utils import check_dir, get_logger
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg, custom_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str, help="folder containing the data (crops)")
    parser.add_argument('--weights-init', type=str,
                        default="")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    parser.add_argument('--bs', type=int, default=32, help='batch_size')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    logger = get_logger(args.output_folder, "KNN")
    data_root = args.data_folder
    # model
    print('this is my device: ', device)
    # build model and load weights
    model = ResNet18Backbone(pretrained=False).to(device)
    # TODO: load weight initialization"
    model.load_state_dict(torch.load(args.weights_init, map_location=device)['model'])
    # dataset
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    # TODO; Load the validation dataset (crops), use the transform above.
    val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    # val_data = DataReaderPlainImg(os.path.join(data_root, str(args.size), "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.bs, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True, collate_fn=custom_collate)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
    #                                          pin_memory=True, drop_last=True, collate_fn=custom_collate)
    # print(val_loader.__dir__())
    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    data_root = os.path.join(data_root, str(args.size), "val")
    query_indices = ["41.jpg"]
    for idx, img in enumerate(val_loader.dataset):
        if val_loader.dataset.image_files[idx] not in query_indices:
            continue
        sample_imagefile = val_loader.dataset.image_files[idx]
        output_path = os.path.join(args.output_folder, sample_imagefile.split(".")[0])
        os.mkdir(output_path)
        print("Computing NNs for sample {}".format(sample_imagefile))
        closest_idx, closest_dist = find_nn(model, img, val_loader, 5)
        closest_image_files = [val_loader.dataset.image_files[idx] for idx in closest_idx]
        logger.info("distances to the closest images are {} ".format(closest_dist))
        logger.info("files of closest images are {} ".format(closest_image_files))
        for nn_file in closest_image_files:
            copyfile(os.path.join(data_root, nn_file), os.path.join(output_path, nn_file))
        copyfile(os.path.join(data_root, sample_imagefile), os.path.join(output_path, sample_imagefile))


def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
        model: the model for computing the features
        query_img: the image of which to find the NNs
        loader: the loader for the dataset in which to look for the NNs
        k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    # TODO: nearest neighbors retrieval
    output = torch.FloatTensor().to(device)
    nn_list = []
    f_rep = model(query_img.unsqueeze(0).to(device))
    for i, image in enumerate(loader.dataset):
        f_rep_i = model(image.unsqueeze(0).to(device))
        # dist = torch.norm((f_rep - f_rep_i), 2)
        dist = ((f_rep - f_rep_i) ** 2).sum(-1)
        if (len(nn_list) >= k+1):
            nn_list.append((i, dist.cpu().item()))
            nn_list.sort(key=lambda nn: nn[1])
            del(nn_list[-1])
        else:
            nn_list.append((i, dist.item()))

    return [x[0] for x in nn_list], [x[1] for x in nn_list]


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
