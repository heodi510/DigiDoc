import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from TextRecognizer.utils import CTCLabelConverter, AttnLabelConverter
from TextRecognizer.dataset import RawDataset, AlignCollate
from TextRecognizer.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
import os
import re
import numpy as np
import math
from statistics import stdev
from pathlib import Path

def load_model(args):

    """ model configuration """
    converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    model = Model(args)
    print('model input parameters', args.imgH, args.imgW, args.num_fiducial, args.input_channel, args.output_channel,
          args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
          args.SequenceModeling)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % args.saved_model)
    model.load_state_dict(torch.load(args.saved_model, map_location=device))
    
    # predict mode
    model.eval()
    
    return args,model,converter

def set_data_loader(args):
    
    aligncollate = AlignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)
    pred_data = RawDataset(root=args.image_folder, opt=args)
    data_loader = torch.utils.data.DataLoader(
        pred_data, batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=aligncollate, pin_memory=True)
    
    return data_loader

def pred_crop_img(args,model,data_loader,converter):

    '''Predict each image in data/crop_img folder and generate a '''
    with torch.no_grad():
        for image_tensors, image_path_list in data_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([args.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, args.batch_max_length + 1).fill_(0).to(device)
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                
                start = str(Path.home())+'/craft/data/'
                path = os.path.relpath(img_name, start)
                image_name=os.path.basename(path)
                file_name='_'.join(image_name.split('_')[:-8])
                
                txt_file=args.output_folder+file_name
                log = open(f'{txt_file}_pred_result.txt', 'a')
                
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                log.write(f'{image_name:25s}\t {pred:25s}\t {confidence_score:0.4f}\n')
            
            log.close()
            
def find_cord(image_name,idx):
    return re.findall(r'\d+\.\d+',image_name)[idx]

def getVerDisStd(df,rm_outlier=True):
    ''' Get standard deviation of consecutive vertical difference '''
    df = df[df['prob']>=0.1]
    df.reset_index(drop=True,inplace=True)
    df['h_level'] = (df['vertex3_y']+df['vertex4_y'])/2
    
    # Series of difference between the consecutive values        
    gaps = [y - x for x, y in zip(df['h_level'][:-1], df['h_level'][1:])]
    std = stdev(gaps)
    if rm_outlier:
        pstl_90 = np.quantile(gaps,0.9)
        gaps_90pstl = [i for i in gaps if i <90]
        std = stdev(gaps_90pstl)

    return std,df

def write_file(df,file_name,output_path):
    
    std,df = getVerDisStd(df)
    
    ### 5. use relative coordinate to to write menu ###
    
    ''' Write menu '''
    list_line = [[df.loc[0,'vertex4_x']]]
    counter=1

    for x in df['h_level'][1:]:
        # if the gap from the current item to the previous is more than 1 SD
        # Note: the previous item is the last item in the last list
        # Note: modify '> 0.6' to adjust threshold for separating lines
        if (x-df.loc[counter-1,'h_level']) / std > 0.6:
            list_line.append([])
        list_line[-1].append(df.loc[counter,'vertex4_x'])
        counter+=1
    
    menu_name=file_name.split('.')[0][:-12]+'_output.txt'
    menu = open(output_path+menu_name,"w") 
    # Decode sentence and write it to menu file
    word_count=0
    for line in list_line:
        # sort words with x coordinate of vertex 4 
        line.sort()
        line_in_preds=[]
        len_sentence = len(line)

        # decode menu content from x coordinate of vertex 4 
        for word in line:
            df_sub_conf = df.loc[word_count:word_count+len_sentence]
            df_sub_conf[df_sub_conf['vertex4_x']==word]['pred'].values[0]
            line_in_preds.append(df_sub_conf[df_sub_conf['vertex4_x']==word]['pred'].values[0])

        # update counter using length of sentence
        word_count+=len_sentence

        # ordered list in prediction word format to menu file
        sentence = ' '.join(line_in_preds)
        menu.write(sentence)
        menu.write('\n')
    menu.close()


def gen_menu(output_folder):
    pred_list = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.endswith('pred_result.txt')]
    for pred_file in pred_list:
        raw_log = pd.read_csv(pred_file,sep="\t", header = None)
        result=raw_log.copy()
        result.columns=['image_name','pred','prob']
        result['vertex1_x']=result.apply(lambda x: find_cord(x.image_name, 0), axis=1).astype(float)
        result['vertex1_y']=result.apply(lambda x: find_cord(x.image_name, 1), axis=1).astype(float)
        result['vertex2_x']=result.apply(lambda x: find_cord(x.image_name, 2), axis=1).astype(float)
        result['vertex2_y']=result.apply(lambda x: find_cord(x.image_name, 3), axis=1).astype(float)
        result['vertex3_x']=result.apply(lambda x: find_cord(x.image_name, 4), axis=1).astype(float)
        result['vertex3_y']=result.apply(lambda x: find_cord(x.image_name, 5), axis=1).astype(float)
        result['vertex4_x']=result.apply(lambda x: find_cord(x.image_name, 6), axis=1).astype(float)
        result['vertex4_y']=result.apply(lambda x: find_cord(x.image_name, 7), axis=1).astype(float)
        result=result[['image_name','vertex1_x','vertex1_y','vertex2_x','vertex2_y','vertex3_x','vertex3_y','vertex4_x','vertex4_y','pred','prob']]
        result.sort_values('vertex4_y',inplace=True)

        # prepare input for write_menu()
        file_name=pred_file.split('/')[-1]
        df=result.copy()
        write_file(df,file_name,output_path)

def gen_rotated_menu(output_folder):
    pred_list = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f)) and f.endswith('pred_result.txt')]
    for pred_file in pred_list:
        raw_log = pd.read_csv(pred_file,sep="\t", header = None)
        result=raw_log.copy()
        result.columns=['image_name','pred','prob']
        result['vertex1_x']=result.apply(lambda x: find_cord(x.image_name, 0), axis=1).astype(float)
        result['vertex1_y']=result.apply(lambda x: find_cord(x.image_name, 1), axis=1).astype(float)
        result['vertex2_x']=result.apply(lambda x: find_cord(x.image_name, 2), axis=1).astype(float)
        result['vertex2_y']=result.apply(lambda x: find_cord(x.image_name, 3), axis=1).astype(float)
        result['vertex3_x']=result.apply(lambda x: find_cord(x.image_name, 4), axis=1).astype(float)
        result['vertex3_y']=result.apply(lambda x: find_cord(x.image_name, 5), axis=1).astype(float)
        result['vertex4_x']=result.apply(lambda x: find_cord(x.image_name, 6), axis=1).astype(float)
        result['vertex4_y']=result.apply(lambda x: find_cord(x.image_name, 7), axis=1).astype(float)
        result.reset_index(inplace=True)
        result['rotation_x']=result['vertex4_x'][0]
        result['rotation_y']=result['vertex4_y'][0]
        result['ratio']=(result['vertex4_y']-result['vertex3_y'])/(result['vertex4_x']-result['vertex3_x'])
        result['theta']=result['ratio'].apply(lambda x: math.atan(x))
        result['theta'].replace(0,np.NaN,inplace=True)
        result['correction_theta']=result['theta'].apply(lambda x:-x).mean()
        result['cos_theta']=result['correction_theta'].apply(lambda x: math.cos(x))
        result['sin_theta']=result['correction_theta'].apply(lambda x: math.sin(x))
        result['new_3x']=(result['vertex3_x']-result['rotation_x'])*result['cos_theta']-(result['vertex3_y']-result['rotation_y'])*result['sin_theta']+ result['rotation_x']
        result['new_3y']=(result['vertex3_x']-result['rotation_x'])*result['sin_theta']+(result['vertex3_y']-result['rotation_y'])*result['cos_theta']+ result['rotation_y']
        result['new_4x']=(result['vertex4_x']-result['rotation_x'])*result['cos_theta']-(result['vertex4_y']-result['rotation_y'])*result['sin_theta']+ result['rotation_x']
        result['new_4y']=(result['vertex4_x']-result['rotation_x'])*result['sin_theta']+(result['vertex4_y']-result['rotation_y'])*result['cos_theta']+ result['rotation_y']
        result['pred']=result['pred'].str.strip()

        result=result[['image_name','vertex1_x','vertex1_y','vertex2_x','vertex2_y','vertex3_x','vertex3_y','vertex4_x','vertex4_y','pred','prob',\
                       'rotation_x','rotation_y','ratio','theta','correction_theta','cos_theta','sin_theta','new_3x','new_3y','new_4x','new_4y']]
        result['pred']=result['pred'].str.strip()
        df=result[(result['prob']<0.3) & (result['pred'].str.contains('\$?\d+', regex=True))]['pred'].apply(lambda x: x[:-1])
        result.update(df)
        result['pred']=result['pred'].apply(lambda x:x.split('-')[0])
        
        result.sort_values('new_4y',inplace=True)
    
        # prepare input for write_menu()
        file_name=pred_file.split('/')[-1]
        df=result.copy()
        write_file(df,file_name,output_path)
    
'''Start of recognizer.py'''  

home = str(Path.home())
weights_path = home+'/craft/TextRecognizer/weights/'
crop_img_path=home+'/craft/data/crop_img/'
output_path = home+'/craft/data/output/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', default=crop_img_path, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default=weights_path+'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', default=True, help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()

    """ vocab / character number configuration """
    if args.sensitive:
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()
    args.output_folder=output_path
    args,recognizer,converter = load_model(args)
    data_loader = set_data_loader(args)
    pred_crop_img(args,recognizer,data_loader, converter)
    gen_menu(args.output_folder)