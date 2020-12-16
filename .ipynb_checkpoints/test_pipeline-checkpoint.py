import sys
import os
import time
import string
import argparse

import CRAFT.pipeline as pipeline
import CRAFT.crop_image as crop_image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np
import CRAFT.craft_utils as craft_utils
import CRAFT.test as test
import CRAFT.imgproc as imgproc
import CRAFT.file_utils as file_utils
from CRAFT.craft import CRAFT

import TextRecognizer.recognizer as recognizer
from TextRecognizer.recognizer import pred_crop_img
from TextRecognizer.recognizer import gen_menu
from TextRecognizer.recognizer import load_model
from TextRecognizer.recognizer import set_data_loader

import json
import zipfile
import pandas as pd
from PIL import Image
from pathlib import Path
from argparse import Namespace
from collections import OrderedDict

home = str(Path.home())
craft_model_path = home+'/craft/CRAFT/weights/'
recog_model_path = home+'/craft/TextRecognizer/weights/'
input_path = home+'/craft/data/input_img/'
result_folder = home+'/craft/data/craft_output/'
crop_img_path=home+'/craft/data/crop_img/'
output_path = home+'/craft/data/output/'
cudnn.benchmark = True
cudnn.deterministic = True

if __name__ =='__main__':
    
    '''prepare args for CRAFT'''
    c_args = Namespace(trained_model=craft_model_path+'craft_mlt_25k.pth', 
                     text_threshold=0.7,
                     low_text=0.4,
                     link_threshold=0.4,
                     cuda=True,
                     canvas_size=1280,
                     mag_ratio=1.5,
                     poly=False,
                     show_time=False,
                     input_folder=input_path,
                     refine=False,
                     refiner_model=craft_model_path+'craft_refiner_CTW1500.pth')
    
    # load craft model
    craft_net = pipeline.load_craft(c_args)

    # For test images in a folder
    image_list, _, _ = file_utils.get_files(c_args.input_folder)
    image_names = []
    
    for num in range(len(image_list)):
        image_names.append(os.path.relpath(image_list[num], c_args.input_folder))

    # create result folder if it is not exist
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    
    c_args.image_list=image_list
    c_args.image_names=image_names
    
    
    
    # load refiner
    if c_args.refine:
        refine_net = pipeline.load_refiner(c_args)
        refine_net.eval()
        c_args.poly = True
    else:
        refine_net = None
    
    '''prepare args for text recognizer'''
    t_args = Namespace(workers=4,
                       batch_size=192,
                       saved_model=recog_model_path+'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth',
                       batch_max_length=25,
                       imgH=32,
                       imgW=100,
                       rgb=None,
                       character=string.printable[:-6], 
                       sensitive=True, 
                       PAD='None', 
                       Transformation='TPS',
                       FeatureExtraction='ResNet',
                       SequenceModeling='BiLSTM',
                       Prediction='Attn', 
                       num_fiducial=20, 
                       input_channel=1, 
                       output_channel=512, 
                       hidden_size=256,
                       num_gpu = torch.cuda.device_count())
    
    t_args,recognizer,converter = load_model(t_args)
    
    # run CRAFT and generate boxed images
    pipeline.run(c_args,craft_net,refine_net)
    # crop word boxes from boxed images
    crop_image.run()
    
    t_args.image_folder=crop_img_path
    t_args.output_folder=output_path
    data_loader = set_data_loader(t_args)
    pred_crop_img(t_args,recognizer,data_loader, converter)
    gen_menu(t_args.output_folder)
    