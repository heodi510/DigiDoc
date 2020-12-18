import os
import shutil
import sys
import time
import cv2
import string
import base64
import pandas as pd
import numpy as np
import streamlit as st


from PIL import Image
from typing import Dict
from pathlib import Path
from argparse import Namespace

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import CRAFT.craft_utils as craft_utils
import CRAFT.test as test
import CRAFT.imgproc as imgproc
import CRAFT.file_utils as file_utils
import CRAFT.pipeline as pipeline
import CRAFT.crop_image as crop_image
from CRAFT.craft import CRAFT

import TextRecognizer.recognizer as recognizer
from TextRecognizer.recognizer import pred_crop_img
from TextRecognizer.recognizer import gen_menu
from TextRecognizer.recognizer import load_model
from TextRecognizer.recognizer import set_data_loader


@st.cache(allow_output_mutation=True)
def setup_CRAFT():
    '''Setup arguments for CRAFT and load pretrained model'''
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
    return c_args,craft_net

@st.cache(allow_output_mutation=True)
def setup_Recognizer():
    '''Setup arguments for text recognizer and load pretrained model'''
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
    
    # load recognizer and converter
    t_args,recognizer,converter = load_model(t_args)
    return t_args,recognizer,converter

def gen_digi_docs(text_menu_list):
    for text in text_menu_list:
        with open(text, "rb") as f:
            text_name=text.split('/')[-1].split('.')[0]
            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            href = f'<a href="data:file/txt;base64,{b64}">Right-click and save as {text_name}.txt</a> '
            st.markdown(href, unsafe_allow_html=True)

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        print('Clear '+file_path)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def clear_all():
    clear_folder(input_path)
    clear_folder(result_folder)
    clear_folder(crop_img_path)
    clear_folder(output_path)
    
            
def main():
    """Run this function to run the app"""
    
    # SIDEBARS
    st.sidebar.header("Navigation")
    st.sidebar.markdown("We help convert menus, documents, invoices or other physical text into an organized CSV file.")
    document_type = st.sidebar.selectbox("Document Type",["Menus","Invoices (Coming Soon)","Tax Forms (Coming Soon)","Contracts (Coming Soon)","Reports (Coming Soon)","Id Documents (Coming Soon)"])
    file_format = st.sidebar.radio("Type of Format",("Format 1","Format 2","Format 3","Format 4"))
    st.sidebar.header("About")
    st.sidebar.info("***Input Your Own Description***")

    # Main Page
    st.image("pic/logo.png",width=600)
    st.title("Change your document to digital files")
    
    result = st.file_uploader("Upload one or more images to convert to CSV", type=["png","jpg","jpeg"],accept_multiple_files=True)

    if result:
        
        # clear all temporary results
        clear_folder(result_folder)
        clear_folder(crop_img_path)
        clear_folder(output_path)
        
        st.info("Total: " + str(len(result)) + " Images")
        
        # save image into input file
        for i,img_file_buffer in enumerate(result):
            img = Image.open(img_file_buffer)
            st.image(img_file_buffer.getvalue(),width=200)
            img.save(input_path+'image_'+str(i+1)+'.jpg')
        
        # ADD MODEL HERE
        if st.button("Run Model"):

            # get image from input folder
            image_list, _, _ = file_utils.get_files(c_args.input_folder)
            image_names = []
            for num in range(len(image_list)):
                image_names.append(os.path.relpath(image_list[num], c_args.input_folder))

            # prepare remaining parameter for comming models
            c_args.image_list=image_list
            c_args.image_names=image_names

            if c_args.refine:
                refine_net = pipeline.load_refiner(c_args)
                refine_net.eval()
                c_args.poly = True
            else:
                refine_net = None

            t_args.image_folder=crop_img_path
            t_args.output_folder=output_path

            # Pipeline for digitalize documents
            pipeline.run(c_args,craft_net,refine_net) # write boxes for every words
            crop_image.run() # crop image for each word
            data_loader = set_data_loader(t_args) # load cropped images to dataloader
            pred_crop_img(t_args,recognizer,data_loader, converter) # predict all cropped images
            gen_menu(t_args.output_folder) # generate txt for each image
            
            # Message for download
            st.markdown('Download txt files')
            text_menu_list = [os.path.join(output_path, f) for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f)) and f.endswith('output.txt')]

            # Generate digital documents
            gen_digi_docs(sorted(text_menu_list))
            
            clear_folder(input_path)
            clear_folder(result_folder)
            clear_folder(crop_img_path)
                

# assign different path
home = str(Path.home())
craft_model_path = home+'/craft/CRAFT/weights/'
recog_model_path = home+'/craft/TextRecognizer/weights/'
input_path = home+'/craft/data/input_img/'
result_folder = home+'/craft/data/craft_output/'
crop_img_path=home+'/craft/data/crop_img/'
output_path = home+'/craft/data/output/'

# Create Global variables
c_args, t_args, craft_net, recognizer, converter = None, None, None, None, None

# load large size parameter and setup 
c_args,craft_net=setup_CRAFT()
t_args,recognizer,converter=setup_Recognizer()

# clear all data folders
clear_all()
# main function
main()



# STYLE = """
# <style>
# img {
#     max-width: 100%;
# }
# </style>
# """


# def main():
#     """Run this function to display the Streamlit app"""
#     st.info(__doc__)
#     st.markdown(STYLE, unsafe_allow_html=True)
#     file = st.file_uploader("Upload file", type=["csv","png","jpg"])
#     show_file = st.empty()

#     if not file:
#         show_file.info("Please Upload a file : {} ".format(' '.join(["csv","png","jpg"])))
#         return
#     content = file.getvalue()

#     # if isinstance(file, BytesIO):
#     #     show_file.image(file)
#     # else:
#     #     df = pd.read_csv(file)
#     #     st.dataframe(df.head(10))
#     file.close()
#     st.image(content)
#     st.text(path.content)

# main()


# # Text/Title
# st.title("Streamlit Tutorials")

# # Header/Subheader
# st.header("This is a header")
# st.subheader("This is a subheader")

# # Text
# st.text("Hello St")

# # Markdown
# st.markdown("### This is a Markdown")

# # Error/Colorful Text
# st.success("Successful")

# st.info("Information")

# st.warning("This is a warning")

# st.error("This is an error Danger")

# st.exception("NameError('name three not defined')")

# # Get Help Info About Python
# st.help(range)


# # Writing Text/Super Fxn
# st.write("text with write")

# st.write(range(10))

# # Images 
# st.image("images/5.jpg",width=300,caption="Menu Image")

# # Videos
# vid_file = open("sample.mp4","rb").read()
# st.video(vid_file)

# # Audio
# # audio_file = open("examplemusic.mp3","rb").read()
# # st.audio(audio_file,format='audio/mp3')


# # Widget
# # Checkbox
# if st.checkbox("Show/Hide"):
#     st.text("Showing or Hiding Widget")


# # Radio Buttons
# status = st.radio("What is your status",("Active","Inactive","Type 3","Type 4"))

# if status == 'Active':
#     st.success("You are Active")
# else:
#     st.warning("Not Active")

# # SelectBox
# occupation = st.selectbox("Your Occupation",["Programmer","Data Scientist","Doctor","Businessman"])
# st.write("You selected this option",occupation)

# # MultiSelect
# location = st.multiselect("Where do you work?",("Central","Taikoo","KwunTong","Causeway","Shatin"))
# st.write("You selected",len(location),'locations')

# # Slider

# level = st.slider("What is your level",1,5)

# # Buttons
# st.button("Simple Button")

# if st.button("About"):
#     st.text("Streamlit is Cool")

# # Text Input
# firstname = st.text_input("Enter Your First Name:")
# if st.button("Submit"):
#     result = firstname.title()
#     st.success(result)

# # Text Area
# message = st.text_area("Enter Your message","Type Here..")
# if st.button("Print"):
#     result = message.title()
#     st.success(result)

# # Date Input
# import datetime
# today = st.date_input("Today is",datetime.datetime.now())

# # Time
# the_time = st.time_input("The time is ",datetime.time())

# # Displaying JSON
# st.text("Display JSON")
# st.json({'name':"Kelvin",'gender':"male"})

# # Display Raw Code
# st.text("Display Raw Code")
# st.code("import numpy as np")

# # Display Raw Code
# with st.echo():
#     # This will also show
#     import pandas as pd
#     df = pd.DataFrame()

# # Progress Bar
# import time
# my_bar = st.progress(0)
# for p in range(10):
#     my_bar.progress(p + 1)

# # Spinner
# with st.spinner("Waiting .."):
#     time.sleep(5)
# st.success("Finished")

# # Balloons
# # st.balloons()

# # SIDEBARS
# st.sidebar.header("About")
# st.sidebar.text("This is Streamlit Tutorial")

# # Functions
# @st.cache
# def run_fxn():
#     return range(100)

# st.write(run_fxn())

# # Plot
# st.pyplot()

# # DataFrames
# st.dataframe(df)

# # Tables
# st.table(df)
