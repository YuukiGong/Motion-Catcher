import os
import sys
sys.path.append('..SEARAFT')
sys.path.append('..SEARAFT/core')
import argparse
import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image,ImageSequence
from config.parser import parse_args
import datasets
from core.raft import RAFT
from core.utils.flow_viz import flow_to_image
from core.utils.utils import load_ckpt


def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)              
    log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
    log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
    heatmap = (log_b * weight).sum(dim=1, keepdim=True)
    return heatmap

def forward_flow(args, model, image1, image2):
    output = model(image1, image2, iters=args.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]
    return flow_final, info_final

def calc_flow(args, model, image1, image2):
    img1 = F.interpolate(image1, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    img2 = F.interpolate(image2, scale_factor=2 ** args.scale, mode='bilinear', align_corners=False)
    H, W = img1.shape[2:]
    flow, info = forward_flow(args, model, img1, img2)
    flow_down = F.interpolate(flow, scale_factor=0.5 ** args.scale, mode='bilinear', align_corners=False) * (0.5 ** args.scale)
    info_down = F.interpolate(info, scale_factor=0.5 ** args.scale, mode='area')
    return flow_down, info_down

@torch.no_grad()
def demo_data(name, args, model, image1, image2, flow_gt):
    path = f"demo/{name}/"
    os.system(f"mkdir -p {path}")
    H, W = image1.shape[2:]
    cv2.imwrite(f"{path}image1.jpg", cv2.cvtColor(image1[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{path}image2.jpg", cv2.cvtColor(image2[0].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
    flow, info = calc_flow(args, model, image1, image2)
    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
    cv2.imwrite(f"{path}flow_final.jpg", flow_vis)
    heatmap = get_heatmap(info, args)
    vis_heatmap(f"{path}heatmap_final.jpg", image1[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())



@torch.no_grad()
def demo_custom(model, args,image1, image2,device=torch.device('cuda')):
    image1 = np.array(image1)
    image2 = np.array(image2)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
    H, W = image1.shape[1:]
    flow_gt = torch.zeros([2, H, W], device=device)
    image1 = image1[None].to(device)
    image2 = image2[None].to(device)
    flow_gt = flow_gt[None].to(device)
    demo_data('custom_downsample', args, model, image1, image2, flow_gt)

@torch.no_grad()
def demo_batch_custom(model, args,image1, image2,device=torch.device('cuda')):
    image1 = torch.stack([torch.from_numpy(arr).permute(2, 0, 1).float().to(device) for arr in image1], dim=0)
    image2 = torch.stack([torch.from_numpy(arr).permute(2, 0, 1).float().to(device) for arr in image2], dim=0)
    # image1 = torch.from_numpy(image1).permute(0,3, 1, 2).float().to(device)
    # image2 = torch.from_numpy(image2).permute(0,3, 1, 2).float().to(device)
    # path = f"demo/batch/"
    # os.system(f"mkdir -p {path}")
    flow, info = calc_flow(args, model, image1, image2)
    
    return flow


def main():
    height = 576
    width = 1024
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,help='cfg',default="../SEARAFT/config/train/Tartan480x640-S.json")
    parser.add_argument('--model', type=str,help='model',default="../SEARAFT/Tartan480x640-S.pth")
    args = parse_args(parser)
    model = RAFT(args)
    load_ckpt(model, args.model)
    model = model.cuda()
    model.eval()
    # gif = Image.open("/data_student_1/gongzhicheng/diffusion/poj/SVD/optical_svd/generated8.gif")
    # imgs = [np.array(frame.copy().convert('RGB')) for frame in ImageSequence.Iterator(gif)]
    
    # demo_custom(model, args, imgs[0].convert('RGB'), imgs[3].convert('RGB'))
    main_folder = "../datasets/around"

    for root, dirs, files in os.walk(main_folder):
        if not dirs:  
            if any(item.endswith(".pt") for item in files):
                continue
            files = [np.array(Image.open(os.path.join(root,frame)).resize((width,height))) for frame in files]
            with torch.no_grad():
                # optical1 = demo_batch_custom(model, args, files[0:13], files[1:14])
                optical1 = demo_batch_custom(model, args, files[0:8], files[1:9])
                optical2 = demo_batch_custom(model, args, files[8:13]+[files[0]], files[9:14]+[files[-1]])
                opticals = torch.cat([optical1, optical2], dim=0)
                torch.save(opticals, os.path.join(root,"optical.pt"))
                print('ok')
            # for file in files:
            #     file_path = os.path.join(root, file)
            #     print("Processing file:", file_path)


if __name__ == '__main__':
    main()

