import copy
import math
from torchvision import datasets, transforms
from torchvision.transforms import ImageOps
from torch.utils.data import ConcatDataset


def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].

    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order
    '''
    if permutation is None:
        return image

    c, h, w = image.size()
    # NOTE: this doesn't preserve the pixels per channel!
    #       (e.g., a pixel from the red channel can end up in the green channel)
    # image = image.view(-1, c)
    # image = image[permutation, :]
    # image = image.view(c, h, w)

    # the code below permutates per channel (same permutation for each channel)
    image = image.view(c, -1)
    image = image[:, permutation]
    image = image.view(c, h, w)

    return image


def _colorize_grayscale_image(image):
    '''Transform [image] from one channel to 3 (identical) channels.'''
    return ImageOps.colorize(image, (0, 0, 0), (255, 255, 255))


def get_dataset(name, train=True, download=True, permutation=None, capacity=None, data_dir='./datasets'):
    data_name = 'mnist' if name=='mnist-color' else name
    dataset_class = AVAILABLE_DATASETS[data_name]
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])
    if data_name=='svhn':
        dataset = dataset_class('{dir}/{name}'.format(dir=data_dir, name=data_name),
                                split="train" if train else "test", download=download, transform=dataset_transform,
                                target_transform=transforms.Compose(AVAILABLE_DATASETS['svhn-target']))
    else:
        dataset = dataset_class('{dir}/{name}'.format(dir=data_dir, name=data_name), train=train,
                                download=download, transform=dataset_transform)

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        return ConcatDataset([
            copy.deepcopy(dataset) for _ in
            range(math.ceil(capacity / len(dataset)))
        ])
    else:
        return dataset


# specify available data-sets.
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'cifar10': datasets.CIFAR10,
    'cifar100': datasets.CIFAR100,
    'svhn': datasets.SVHN,
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'mnist-color': [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Lambda(lambda x: _colorize_grayscale_image(x)),
        transforms.Pad(2),
        transforms.ToTensor(),
    ],
    'cifar10': [
        transforms.ToTensor(),
    ],
    'cifar100': [
        transforms.ToTensor(),
    ],
    'svhn': [
        transforms.ToTensor(),
    ],
    'svhn-target': [
        transforms.Lambda(lambda y: y % 10),
    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10},
    'mnist-color': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'cifar100': {'size': 32, 'channels': 3, 'classes': 100},
    'svhn': {'size': 32, 'channels': 3, 'classes': 10},
}
