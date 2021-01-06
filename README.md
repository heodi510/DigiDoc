# DigiDoc
<h1 align="center">
  <br>
  <img src="https://github.com/heodi510/DigiDoc/blob/main/pic/logo.png" alt="DigiDoc" width="400">
</h1>
<h4 align="center">Helper for transforming your hardcopy document to softcopy!</h4>

DigiDoc is a one-stop text digitization solution for businesses which can convert documents, invoices or other physical text into an organised tabular format using combined machine learning models. 

It helps simplify the data entry process and reduce manual workload, thereby improving business costs and efficiency. 
Users simply input an image of the document and through our web application, can transform and download it as a txt or csv file. 

Our first step towards DigiDoc is to digitize restaurant menus smoothly for restaurant owners and online food delivery services. 
And then slowly transition towards other physical docucments such as contracts and application forms. We've deployed our model onto Streamlit, try us out!

# Demo Preview
![](pic/DigiDoc_demo.gif)

This gif shows a quick demo for our project
1. Drap and drop images to DigiDoc
2. The uploaded images are shown
3. Choose output file format (default is csv)
4. Click 'Run Model' button
5. Preview of the output files are shown
6. Download the output file with links provided

Remarks: DigiDoc is targeting image menu at this moment. 
With more data and further development, it will supoort for other document in images such as invoice, business forms, financial report, etc.

# Table of contents
<!--ts-->
   * [DigiDoc](#DigiDoc)
   * [Demo Preview](#Demo-Preview)
   * [Table of contents](#table-of-contents)
   * [Prerequisite](#prerequisite)
   * [Project Structure](#project-structure)
      * [CRAFT](#craft)
      * [Text Recognizer](#test-recognizer)
      * [Sequential Writer](#sequential)
      * [CSV Converter](#csv-converter)
      * [Web Application](#web-application)
   * [Development](#development)
      * [Data Collection](#data-collection)
      * [Training & Evaluation](#training-&-evaluation)

   * [Deployment](#deployment)
   * [Result](#result)
      * [Performance](#preformance)
      * [Challenge](#challenge)
      * [Improvement](#improvement)
   * [Reference](#reference)
   * [Licensing](#licensing)
<!--te-->

# Prerequisite

1. Git clone this project

      ```git clone https://github.com/heodi510/DigiDoc.git```
      
2. Install packages indicated inside requirements.txt

      ```pip install requirements.txt```

3. Create data folders as below
      ```
      DigiDoc
      └──data
        ├── craft_output
        ├── crop_img
        ├── input_img
        └── output
      ```
4. Download pretrained model from clovaai/CRAFT-pytorch

    *Resource* | *Location* | *Link* 
    ------------- |--------- |---------
    **CRAFT-Pytorch Repository** | -- | [Click](https://github.com/clovaai/CRAFT-pytorch)
    **CRAFT for General Purpose** | CRAFT/weights | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
    **TextRcognizer** | TextRecognizer/weights |[click](https://drive.google.com/file/d/1Z9cof1b6F-GONooOesPE2vQC-YzjvAY3/view?usp=sharing)
    
# Project Structure

The basic structure of DigiDoc is a pipeline of modules. Each module perform a different function on user input and process the input to next module.
Here is the pipline of modeule:

![](pic/model_structure.jpg)

**CRAFT**

The CRAFT model is a VGG16 based model which calculate the similiarity of text and group the characters as a word. 
For each grouped characters, the model will draw a smallest rectangle and bound the whole word. 
Once we have the rectangular boundary, we found out 4 vertexs of the rectangle and cropped it with openCV library. 
The cropped images are passed to text recognizer.

**Text Recognizer**

The text recognizer is a compounded model containing 4 modules as show above.
The TPS module is for transform the input and make it more likely as a proper image with normal angle.
The ResNet Feature Extractor is a ResNet based model without the toppest classification layers.
It is responsible for founding the image features of text image and return the feature vectors with multiple dimensions.
The BiLSTM and Attention layer is to predict the next character. 
With the help the image features of the word, the model can give a high performance prediction of the word based on both computer vision and NLP approach.

**Sequential Writer**

The sequential writer is a function which order all the prediction from text recognizer correctly and write it as text or csv file.
For every single word, the coordinates of bottom vertices are used to calculate a 'vertical level' and seperate each words in different line.
The x coordinate of bottom vertices are used to sorted the word in each line with correct order. Each line with sorted words is combined to single text file
and it is ready to export to user or pass to NER model for generating csv file.

**NER**

The NER model is trained using NLTK. 
With tag of POS and finding out the location of dollar sign, we classifiy the information in text file as a price or a dish discription. 
Finally export the data in csv format.  

# Development

Since the project is built within less than a month, the procedure involved is based on simple machine learning life cycle. It is not designed to support CI/CD and online training as business case nowadays.

**Data Collection**

Two different kinds of data is collected. One is image data for text recognization and the other one is New York Dataset for menu image

For text recognization:\n
training dataset: [Synthetic Word Dataset](https://www.robots.ox.ac.uk/~vgg/data/text/), [Focused Scene Text](https://rrc.cvc.uab.es/?ch=2)\n
validation dataset: [Incidental Scene Text](https://rrc.cvc.uab.es/?ch=4), [IIIT 5K](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)\n
test dataset: [The Street View Text Dataset](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)

For NER model:\n 
For menu image and text content for NER:\n
[Menu data for New York Public Library](http://menus.nypl.org/menus)

**Training & Evaluation**

Hardware config:\n
Machine typen: 1-standard-8 (8 vCPUs, 30 GB memory)\n
GPUs: 1 x NVIDIA Tesla V100

Training Time:\n
Around 22.6 hrs

Evaluation Metrics for text recognization:\n
Categorical Cross Entropy (Loss function)\n
Accuracy, Normalised Edit Distance 

# Deployment

**Steamlit**\n
Framework of web application

**Google App Engine**
GCP tool for deployment

# Result


**Performance**
1. Good performance on text prediction since the training data has tons of scene text.
2. Higher error on numbers as training data contain less image of arabic numbers comparing to scene text image.
image_of_number_error
3. The text generated by sequential writeris not well contructed due to different styles of menu.
Sometimes it is difficult for the sequential writer to distinguish which word belongs to which sentence. 
4. Poor performance on generating csv if the content in text file is not well structured
5. Do not support multiple users simultaneously as we didn't design complicated backend to handle different session or threads for each single users.

**Possible Improvement**

# Reference

[1] M. Jaderberg, K. Simonyan, A. Vedaldi, and A. Zisserman. Synthetic data and artificial neural networks for natural scenetext recognition. In Workshop on Deep Learning, NIPS, 2014. 
[2] A. Gupta, A. Vedaldi, and A. Zisserman. Synthetic data fortext localisation in natural images. In CVPR, 2016. 
[3] D. Karatzas, F. Shafait, S. Uchida, M. Iwamura, L. G. i Big-orda, S. R. Mestre, J. Mas, D. F. Mota, J. A. Almazan, andL. P. De Las Heras. ICDAR 2013 robust reading competition. In ICDAR, pages 1484–1493, 2013. 
[4] D. Karatzas, L. Gomez-Bigorda, A. Nicolaou, S. Ghosh, A. Bagdanov, M. Iwamura, J. Matas, L. Neumann, V. R.Chandrasekhar, S. Lu, et al. ICDAR 2015 competition on ro-bust reading. In ICDAR, pages 1156–1160, 2015. 
[5] A. Mishra, K. Alahari, and C. Jawahar. Scene text recognition using higher order language priors. In BMVC, 2012. 

# Licensing
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
