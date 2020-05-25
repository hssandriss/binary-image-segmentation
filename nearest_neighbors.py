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
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                             pin_memory=True, drop_last=True, collate_fn=custom_collate)
    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.

    query_indices = ["41.jpg", "53.jpg", "55.jpg", "80.jpg"]
    print(data_root)
    for idx, img in enumerate(val_loader.dataset):
        if val_loader.dataset.image_files[idx] not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist = find_nn(model, img, val_loader, 5, logger)
        # TODO: retrieve the original NN images, save them and log the results."
        sample_imagefile = val_loader.dataset.image_files[idx]
        closest_image_files = [val_loader.dataset.image_files[idx] for idx in closest_idx.tolist()]
        logger.info("distances to the closest images are {} ".format(closest_dist.tolist()))
        logger.info("files of closest images are {} ".format(closest_image_files))
        for nn_file in closest_image_files:
            copyfile("./%s/256/val/%s" % (args.data_root, nn_file), "./%s/%i/%s" %
                     (args.output_folder, sample_imagefile.split(".")[0], nn_file))
        copyfile("./%s/256/val/%s" % (args.data_root, sample_imagefile), "./%s/%i/%s" %
                 (args.output_folder, sample_imagefile.split(".")[0], sample_imagefile))


def find_nn(model, query_img, loader, k, logger):
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
    distances = torch.FloatTensor().to(device)
    f_rep = model(query_img.unsqueeze(0).to(device))
    for i, image in enumerate(loader.dataset):
        f_rep_i = model(image.unsqueeze(0).to(device))
        # dist = torch.norm((f_rep - f_rep_i), 2)
        dist = ((f_rep - f_rep_i) ** 2).sum(-1)
        distances = torch.cat((distances, dist))
        if i % 100:
            logger.info("iter {}/{} ".format(i/100, len(loader.dataset)/100))

    closest_dist, closest_idx = distances.topk(k)
    return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
