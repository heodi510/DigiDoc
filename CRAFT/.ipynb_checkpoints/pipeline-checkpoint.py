import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from pathlib import Path
from PIL import Image
from skimage import io

import cv2
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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def load_craft(args):
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(test.copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(test.copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    net.eval()
    
    return net

def load_refiner(args):
    from CRAFT.refinenet import RefineNet
    refine_net = RefineNet()
    print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
    if args.cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
    return refine_net


def run(args,model,refiner):

    data=pd.DataFrame(columns=['image_name', 'word_bboxes', 'pred_words', 'align_text'])
    data['image_name'] = args.image_names

    # Craft
    net = model

    # LinkRefiner
    refine_net =refiner

    t = time.time()

    # load data
    for k, image_path in enumerate(args.image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(args.image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, det_scores = test.test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args, refine_net)
        
        bbox_score={}

        for box_num in range(len(bboxes)):
            key = str (det_scores[box_num])
            item = bboxes[box_num]
            bbox_score[key]=item

        data['word_bboxes'][k]=bbox_score
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    data.to_csv(result_folder+'data.csv', sep = ',', na_rep='Unknown')
    print("elapsed time : {}s".format(time.time() - t))
    

# initalize paths
# modify the absolute path if necessary
home = str(Path.home())
weights_path = home+'/DigiDoc/CRAFT/weights/'
input_path = home+'/DigiDoc/data/input_img/'
result_folder = home+'/DigiDoc/data/craft_output/'
    
if __name__ == '__main__':
    
    # create argument parser
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default=weights_path+'craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--input_folder', default=input_path, type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default=weights_path+'craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
    args = parser.parse_args()


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
    craft_net = load_craft(args)
    
    # load refiner
    if args.refine:
        refine_net = load_refiner(args)
        refine_net.eval()
        args.poly = True
    else:
        refine_net = None
    run(args,craft_net,refine_net)
    