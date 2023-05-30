import streamlit as st
from PIL import Image

import os
import argparse

import numpy as np

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

st.markdown('# 在线动漫化网站')
uploaded_file = st.file_uploader("请选择需要动漫化的图片",type="jpg")

if uploaded_file is not None:
    in_image = Image.open(uploaded_file).convert("RGB")
    out_image = in_image.copy()
    def load_image(img, x32=False):
        if x32:
            def to_32s(x):
                return 256 if x < 256 else x - x % 32
            w, h = img.size
            img = img.resize((to_32s(w), to_32s(h)))

        return img

    def test(args):
        device = args.device
    
        net = Generator()
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        net.to(device).eval()
        image = load_image(in_image, args.x32)

        with torch.no_grad():
            image = to_tensor(image).unsqueeze(0) * 2 - 1
            out = net(image.to(device), args.upsample_align).cpu()
            out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = to_pil_image(out)
            out.save("x.jpg") 


    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--checkpoint',
            type=str,
            default='./weights/face_paint_512_v2.pt',
        )
        parser.add_argument(
            '--input_dir', 
            type=str, 
            default='./samples/inputs',
        )
        parser.add_argument(
            '--output_dir', 
            type=str, 
            default='./samples/results',
        )
        parser.add_argument(
            '--device',
            type=str,
            default='cuda:0',
        )
        parser.add_argument(
            '--upsample_align',
            type=bool,
            default=False,
            help="Align corners in decoder upsampling layers"
        )
        parser.add_argument(
            '--x32',
            action="store_true",
            help="Resize images to multiple of 32"
        )
        args = parser.parse_args()
    
        test(args)

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption='原图片')
    with col2:
        st.image("x.jpg", caption='动漫化图片')

