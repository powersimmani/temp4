import matplotlib.pyplot as plt
import numpy as np
import torch, code,copy, torchvision
from torchvision import transforms
from cifar_dataset import cifar10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def imshow(img, title = "", path = None):
	#img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.title(title+" ")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    if path is not None:
    	plt.savefig(path)
    plt.close()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
#Original image
index = 20
path = "aug_images/"

image = trainset[index][0] / 2 + 0.5
imshow(image, 'original',path+"0.original.png")

#nomalized image
image = trainset[index][0]
imshow(image, 'normalized',path+"1.normalized.png")

#Jitter image
trans_jitter = transforms.Compose([transforms.ToPILImage(),
									transforms.ColorJitter(brightness=3, contrast=0, saturation=0, hue=0),
									transforms.ToTensor()])
image = trainset[index][0] / 2 + 0.5
image = trans_jitter(image)
imshow(image, 'jittered',path+"2.jittered.png")

#Affine image
trans_affine = transforms.Compose([transforms.ToPILImage(),
									transforms.RandomAffine(30, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
									transforms.ToTensor()])
image = trainset[index][0] / 2 + 0.5
image = trans_affine(image)
imshow(image, 'affined',path + "3.affined.png")


#Random crop
trans_crop = transforms.Compose([transforms.ToPILImage(),
									transforms.RandomCrop((16,16), padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
									transforms.Resize((32,32)),
									transforms.ToTensor()])

image = trainset[index][0] / 2 + 0.5
image = trans_crop(image)
imshow(image, 'cropped',path + "4.cropped.png")


#Vertial flip
trans_flip_ver = transforms.Compose([transforms.ToPILImage(),
									transforms.RandomVerticalFlip(p=1),
									transforms.Resize((32,32)),
									transforms.ToTensor()])

image = trainset[index][0] / 2 + 0.5
image = trans_flip_ver(image)
imshow(image, 'vertial_flip',path + "5.vertial_flip.png")

#Horizontal flip
trans_flip_hor = transforms.Compose([transforms.ToPILImage(),
									transforms.RandomHorizontalFlip(p=1),
									transforms.Resize((32,32)),
									transforms.ToTensor()])

image = trainset[index][0] / 2 + 0.5
image = trans_flip_hor(image)
imshow(image, 'hor_flip',path + "6.hor_flip.png")
code.interact(local=dict(globals(), **locals()))

