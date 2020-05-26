from models.pretraining_backbone import ResNet18Backbone
from models.second_segmentation import Segmentator
from data.transforms import get_transforms_binary_segmentation
from utils import check_dir, get_logger
import argparse
from PIL import Image
import os
import torch
from torchvision.transforms import *
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Choose model (binary, multiclass, att)")
    parser.add_argument('--weights-init', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    output = "./examples"
    root = "./examples/samples"
    for entry in os.scandir(root):
        if (entry.path.endswith(".jpg")):
            file_name = entry
            img = Image.open(os.path.join(entry))
            transforms = Compose([ToTensor()])
            # transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
            img = transforms(img)
            img = img.unsqueeze(0).cuda()
            if args.model == "binary":
                pretrained_model = ResNet18Backbone(pretrained=False)
                model = Segmentator(2, pretrained_model.features).cuda()
                model.load_state_dict(torch.load(args.weights_init)['model'])
                model.eval()
                with torch.no_grad():
                    label = model(img)
                    print(label.shape)
                    label = torch.argmax(label, dim=1, keepdim=True).float()
                label = label.squeeze(0).cpu()
                im = ToPILImage()(label)
                im = im.convert('RGB')
                im.show()
                im.save(os.path.join(output, "{}_binseg.jpg".format(file_name.name.split('.')[0])))
            elif args.model == "multiclass":
                pretrained_model = ResNet18Backbone(pretrained=False)
                model = Segmentator(6, pretrained_model.features).cuda()
                model.load_state_dict(torch.load(args.weights_init)['model'])
                model.eval()
                with torch.no_grad():
                    label = model(img)
                    print(label.shape)
                    label = torch.argmax(label, dim=1, keepdim=True).float()
                    print(label.shape)
                for c in range(label.shape[1]):
                    im = label[:, c, :, :].squeeze(0).cpu()
                    im = ToPILImage()(im)
                    im = im.convert('RGB')
                    im.show()
                    im.save(os.path.join(output, "{}_multiseg_class_.jpg".format(file_name.name.split('.')[0], c)))
            elif args.model == "multiclass":
                pass
