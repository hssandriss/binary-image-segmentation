# from data.transforms import ImgRotation
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from data.transforms import ImgRotation
# Read the image from file. Assuming it is in the same directory.
pil_image = Image.open('./data/0.jpg')
pil2tensor = transforms.ToTensor()
# Plot the image here using matplotlib.
angles = [0, 90, 180, 270]


def plot_image(tensor):
    plt.figure()
    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.show()


rotated_imgs = []
labels = []
for label, angle in enumerate(angles):
    # rotated_imgs.append(TF.rotate(pil_image, angle))
    rotated_imgs.append(pil_image.rotate(angle))
    labels.append(label)
    print(label)
    plot_image(pil2tensor(rotated_imgs[-1]))
