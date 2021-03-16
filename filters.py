import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19


# gray scale
def bgr2gray(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


# Canny edge detection
def canny_edge(img):
    canny_edges = cv2.Canny(img, 100, 200)
    return canny_edges


# Gabor filter
def gabor_filter(k_size=111, sigma=10, gamma=1.2, g_lambda=10, psi=0, angle=0):
    # get half size
    d = k_size // 2

    # prepare kernel
    gabor = np.zeros((k_size, k_size), dtype=np.float32)

    # each value
    for y in range(k_size):
        for x in range(k_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x ** 2 + gamma ** 2 * _y ** 2) / (2 * sigma ** 2)) * np.cos(
                2 * np.pi * _x / g_lambda + psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


# Use Gabor filter to act on the image
def gabor_filtering(gray, k_size=111, sigma=10, gamma=1.2, g_lambda=10, psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (k_size // 2, k_size // 2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = gabor_filter(k_size=k_size, sigma=sigma, gamma=gamma, g_lambda=g_lambda, psi=0, angle=angle)

    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y: y + k_size, x: x + k_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Use 6 Gabor filters with different angles to perform feature extraction on the image
def gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = bgr2gray(img).astype(np.float32)

    # define angle
    # As = [0, 45, 90, 135]
    As = [0, 30, 60, 90, 120, 150]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = gabor_filtering(gray, k_size=9, sigma=1.5, gamma=1.2, g_lambda=1, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


class VGG19:
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = vgg19(pretrained=True)
        self.model.eval()
        # remove last fully connected layer
        self.model.classifier = torch.nn.Sequential(*list(self.model.classifier.children())[:-1])

    def forward(self, img):
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            features = self.model(input_batch)
        return features

# Read image
# img = cv2.imread('dataset/COVID/COVID (1).PNG').astype(np.float32)

# gabor process
# out = gabor_process(img)

# canny edge process
# out = canny_edge(img)

# image = Image.open('datasets/covid19/train/COVID/COVID (1).png').convert("RGB")
# vgg = VGG19()
# features = vgg.forward(image)
# print(features)
