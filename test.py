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
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--self-ensemble', type=bool, default=False)
    args = parser.parse_args()
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net(scale_factor=args.scale,num_channels=args.num_channels,num_features=args.num_features,growth_rate=args.growth_rate,B=args.B, U=args.U).to(device)
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
    psnr_average=0
    img_process_time=0
    print("----------Urban100 start----------")
    for i in range(1, 101):
        if i<=9:
            image_file="data/Urban100/x{}/img_00{}_SRF_{}_HR.png".format(args.scale,i,args.scale)
        elif i>9 and i<=99:
            image_file="data/Urban100/x{}/img_0{}_SRF_{}_HR.png".format(args.scale,i,args.scale)
        
        elif i==100:
            image_file="data/Urban100/x{}/img_100_SRF_{}_HR.png".format(args.scale,args.scale)
                
        image = pil_image.open(image_file).convert('RGB')
		image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
		hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        bicubic.save(image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
		lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        lr = torch.from_numpy(lr).to(device)
        hr = torch.from_numpy(hr).to(device)
		if args.self_ensemble:
            with torch.no_grad():
                Urban100_start = time.time()  # 시작 시간 저장
                preds = x8_forward(lr,model).squeeze(0)
                Urban100_img_process_ex=time.time()-Urban100_start
        else :
            with torch.no_grad():
                Urban100_start = time.time()  # 시작 시간 저장
                preds = model(lr).squeeze(0)
                Urban100_img_process_ex=time.time()-Urban100_start
        preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
        hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')
		preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
        hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]
		psnr = calc_psnr(hr_y, preds_y)
        print('{}..PSNR: {:.6f}'.format(i, psnr))
        print('Urban100_img_process_time : {:.2f}'.format(Urban100_img_process_ex))
        img_process_time += Urban100_img_process_ex
        psnr_average=psnr_average+float(psnr)
		output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
        output.save(image_file.replace('.', '_RDCAB_x{}.'.format(args.scale)))
	Urban100_avg = psnr_average/100
    Urban100_avg_time = img_process_time/100
    print("----------Urban100 End----------")
    psnr_average=0
    img_process_time=0
    print("----------BSD100 start----------")
    for i in range(1, 101):
        if i<=9:
            image_file="data/BSD100/x{}/img_00{}_SRF_{}_HR.png".format(args.scale,i,args.scale)
        elif i>9 and i<=99:
            image_file="data/BSD100/x{}/img_0{}_SRF_{}_HR.png".format(args.scale,i,args.scale)
        
        elif i==100:
            image_file="data/BSD100/x{}/img_100_SRF_{}_HR.png".format(args.scale,args.scale)
                
        image = pil_image.open(image_file).convert('RGB')
		image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
		hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        bicubic.save(image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
		lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        lr = torch.from_numpy(lr).to(device)
        hr = torch.from_numpy(hr).to(device)
		if args.self_ensemble:
            with torch.no_grad():
                BSD100_start = time.time()  # 시작 시간 저장
                preds = x8_forward(lr,model).squeeze(0)
                BSD100_img_process_ex=time.time()-BSD100_start
        else :
            with torch.no_grad():
                BSD100_start = time.time()  # 시작 시간 저장
                preds = model(lr).squeeze(0)
                BSD100_img_process_ex=time.time()-BSD100_start
        
        preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
        hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')
		preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
        hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]
		psnr = calc_psnr(hr_y, preds_y)
        print('{}..PSNR: {:.2f}'.format(i, psnr))
		print('BSD100_img_process_time : {:.6f}'.format(BSD100_img_process_ex))
        img_process_time += BSD100_img_process_ex
        psnr_average=psnr_average+float(psnr)
		output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
        output.save(image_file.replace('.', '_RDCAB_x{}.'.format(args.scale)))
	BSD100_avg = psnr_average/100 
    BSD100_avg_time = img_process_time/100
    print("----------BSD100 End----------")
    psnr_average = 0
    img_process_time = 0
    print("----------Set14 start----------")
    for i in range(1, 15):
		if i<=9:
            image_file="data/Set14/x{}/img_00{}_SRF_{}_HR.png".format(args.scale,i,args.scale)
        elif i>9 and i<=14:
            image_file="data/Set14/x{}/img_0{}_SRF_{}_HR.png".format(args.scale,i,args.scale)
		image = pil_image.open(image_file).convert('RGB')
		image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
		hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        bicubic.save(image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
		lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        lr = torch.from_numpy(lr).to(device)
        hr = torch.from_numpy(hr).to(device)
		if args.self_ensemble:
            with torch.no_grad():
                Set14_start = time.time()  # 시작 시간 저장
                preds = x8_forward(lr,model).squeeze(0)
                Set14_img_process_ex=time.time()-Set14_start
        else :
            with torch.no_grad():
                Set14_start = time.time()  # 시작 시간 저장
                preds = model(lr).squeeze(0)
                Set14_img_process_ex=time.time()-Set14_start
        preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
        hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')
		preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
        hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]
		psnr = calc_psnr(hr_y, preds_y)
        print('{}..PSNR: {:.2f}'.format(i, psnr))
        print('Set14_img_process_time : {:.6f}'.format(Set14_img_process_ex))
        img_process_time += Set14_img_process_ex
        psnr_average=psnr_average+float(psnr)
		output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
        output.save(image_file.replace('.', '_RDCAB_x{}.'.format(args.scale)))
	Set14_avg = psnr_average/14
    Set14_avg_time = img_process_time/14
    print("----------Set14 End----------")
    psnr_average = 0 
    img_process_time =0
    print("----------Set5 start----------")
    for i in range(1, 6):
        image_file="data/Set5/x{}/img_00{}_SRF_{}_HR.png".format(args.scale,i,args.scale)
        image = pil_image.open(image_file).convert('RGB')
		image_width = (image.width // args.scale) * args.scale
        image_height = (image.height // args.scale) * args.scale
		hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
        bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        bicubic.save(image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))
		lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        lr = torch.from_numpy(lr).to(device)
        hr = torch.from_numpy(hr).to(device)
		if args.self_ensemble:
            with torch.no_grad():
                Set5_start = time.time()  # 시작 시간 저장
                preds = x8_forward(lr,model).squeeze(0)
                Set5_img_process_ex=time.time()-Set5_start
        else :
            with torch.no_grad():
                Set5_start = time.time()  # 시작 시간 저장
                preds = model(lr).squeeze(0)
                Set5_img_process_ex=time.time()-Set5_start
        preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
        hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')
		preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
        hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]
		psnr = calc_psnr(hr_y, preds_y)
        print('{}..PSNR: {:.2f}'.format(i, psnr))
        Set5_img_process_ex=time.time()-Set5_start
        print('Set5_img_process_time : {:.6f}'.format(Set5_img_process_ex))
        img_process_time += Set5_img_process_ex
        psnr_average=psnr_average+float(psnr)
		output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
        output.save(image_file.replace('.', '_RDCAB_x{}.'.format(args.scale)))
	Set5_avg = psnr_average/5
    Set5_avg_time = img_process_time/5
    print("----------Set5 End----------")
    psnr_average = 0
    img_process_time = 0
    print('Set5 average psnr : %0.2f' %(Set5_avg))
    print('Set5 average time : %0.6f' %(Set5_avg_time))
    print('Set14 average psnr : %0.2f' %(Set14_avg))
    print('Set14 average time : %0.6f' %(Set14_avg_time))
    print('BSD100 average psnr : %0.2f' %(BSD100_avg))
    print('BSD100 average time : %0.6f' %(BSD100_avg_time))
    print('Urban100 average psnr : %0.2f' %(Urban100_avg))
    print('Urban100 average time : %0.6f' %(Urban100_avg_time))
