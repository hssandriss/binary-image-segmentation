import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-init', type=str,
                        default="")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    raise NotImplementedError("TODO: build model and load weights snapshot")

    # dataset
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    raise NotImplementedError("Load the validation dataset (crops), use the transform above.")

    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = []
    nns = []
    for idx, img in enumerate(val_loader):
        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist = find_nn(model, img, val_loader, 5)
        raise NotImplementedError("TODO: retrieve the original NN images, save them and log the results.")


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
    raise NotImplementedError("TODO: nearest neighbors retrieval")
    # return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
