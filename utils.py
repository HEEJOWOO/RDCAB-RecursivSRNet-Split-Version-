import torch
import numpy as np


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()
def PSNR(a, b, max=255.0, shave_border=0):
    assert type(a) == type(b)
    assert (type(a) == torch.Tensor) or (type(a) == np.ndarray)

    a = a[shave_border:a.shape[0]-shave_border, shave_border:a.shape[1]-shave_border]
    b = b[shave_border:b.shape[0]-shave_border, shave_border:b.shape[1]-shave_border]

    if type(a) == torch.Tensor:
        return 10. * ((max ** 2) / ((a - b) ** 2).mean()).log10()
    elif type(a) == np.ndarray:
        return 10. * np.log10((max ** 2) / np.mean(((a - b) ** 2)))
    else:
        raise Exception('The PSNR function supports torch.Tensor or np.ndarray types.', type(a))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_weights(model, path):
    state_dict = model.state_dict()
    for n, p in torch.load(path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
    return model