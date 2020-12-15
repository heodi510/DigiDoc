import sys
import os
import time
import argparse
from pathlib import Path
from argparse import Namespace
import CRAFT.pipeline as pipeline
import CRAFT.crop_image as crop_image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import CRAFT.craft_utils as craft_utils
import CRAFT.test as test
import CRAFT.imgproc as imgproc
import CRAFT.file_utils as file_utils
import json
import zipfile
import pandas as pd

from CRAFT.craft import CRAFT

from collections import OrderedDict


home = str(Path.home())
weights_path = home+'/craft/CRAFT/weights/'
input_path = home+'/craft/data/input_img/'
result_folder = home+'/craft/data/craft_output/'

if __name__ =='__main__':
    args = Namespace(trained_model=weights_path+'craft_mlt_25k.pth', 
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
                     refiner_model=weights_path+'craft_refiner_CTW1500.pth')
    
    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.input_folder)
    image_names = []
    
    for num in range(len(image_list)):
        image_names.append(os.path.relpath(image_list[num], args.input_folder))

    # create result folder if it is not exist
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    
    args.image_list=image_list
    args.image_names=image_names
    
    # load craft model
    craft_net = pipeline.load_craft(args)
    
    # load refiner
    if args.refine:
        refine_net = pipeline.load_refiner(args)
        refine_net.eval()
        args.poly = True
    else:
        refine_net = None
    
    pipeline.run(args,craft_net,refine_net)
    crop_image.run()