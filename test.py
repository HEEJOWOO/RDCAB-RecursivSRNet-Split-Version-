import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import Net
from utils import convert_rgb_to_y, denormalize, calc_psnr
import time

#test : Set5, Set14, BSD100, Urban100
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=1)
    parser.add_argument('--U', type=int, default=9)
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--num-features', type=int, default=64)
    parser.add_argument('--growth-rate', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--self-ensemble', type=bool, default=False)
    args = parser.parse_args()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net(scale_factor=args.scale,num_channels=args.num_channels,num_features=args.num_features,growth_rate=args.growth_rate,num_layers=args.num_layers,B=args.B, U=args.U).to(device)
    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
##############################################
##Self-Ensemble
##############################################
    def x8_forward(img, model, precision='single'):
        def _transform(v, op):
            if precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'vflip':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'hflip':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 'transpose':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()
            
            ret = torch.Tensor(tfnp).cuda()

            if precision == 'half':
                ret = ret.half()
            elif precision == 'double':
                ret = ret.double()

            with torch.no_grad():
                ret = Variable(ret)

            return ret

        inputlist = [img]
        for tf in 'vflip', 'hflip', 'transpose':
            inputlist.extend([_transform(t, tf) for t in inputlist])

        outputlist = [model(aug) for aug in inputlist]
        for i in range(len(outputlist)):
            if i > 3:
                outputlist[i] = _transform(outputlist[i], 'transpose')
            if i % 4 > 1:
                outputlist[i] = _transform(outputlist[i], 'hflip')
            if (i % 4) % 2 == 1:
                outputlist[i] = _transform(outputlist[i], 'vflip')
        
        output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

        return output
##############################################            
    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    lr = torch.from_numpy(lr).to(device)
    hr = torch.from_numpy(hr).to(device)

    if args.self_ensemble:
        with torch.no_grad():
	    preds = x8_forward(lr,model).squeeze(0)
    else :
        with torch.no_grad():
	    preds = model(lr).squeeze(0)
	
    preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
    hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

    preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
    hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

    psnr = calc_psnr(hr_y, preds_y)
    print('PSNR: {:.2f}'.format(psnr))

    output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
    output.save(args.image_file.replace('.', '_rdn_x{}.'.format(args.scale)))
