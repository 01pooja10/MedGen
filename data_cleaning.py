#imports
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.applications.densenet import DenseNet121,preprocess_input
from keras.layers import Activation,Flatten,Dense
import keras.backend as K
import PIL
from PIL import Image
import shutil
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import seaborn as sns
import string
import nltk
from nltk.tokenize import word_tokenize
import pickle
import re

#unzipping data folders
!unzip '/content/drive/My Drive/chest_xrays.zip'
img=cv2.imread('/content/chest_xrays/CXR999_IM-2480-3001.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(img)
#os.listdir('/content')
!unzip '/content/drive/My Drive/reports.zip'

#extract the respective reports from xml documents
img = []
img_impression = []
img_finding = []
#assign dir
directory = '/content/ecgen-radiology'
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".xml"):
        f = directory + '/' + filename
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if child.tag == 'MedlineCitation':
                for attr in child:
                    if attr.tag == 'Article':
                        for i in attr:
                            if i.tag == 'Abstract':
                                for name in i:
                                    if name.get('Label') == 'FINDINGS':
                                        finding=name.text
        for p_image in root.findall('parentImage'):
            img.append(p_image.get('id'))
            img_finding.append(finding)

#create dataframe
df=pd.DataFrame()
df['image_id']=img
df['reports']=img_finding
df.head()
#df.isna().sum()
df.dropna(axis=0,inplace=True)
df['human_id']=df['image_id'].apply(lambda x: x.strip().split('-')[0])+'-'+df['image_id'].apply(lambda x: x.strip().split('-')[1])
#df.head()

#converting to jpg
for name in df['image_id'].values:

    img=Image.open('/content/chest_xrays/'+name+'.png').resize((224,224)).convert('RGB')
    img.save('/content/chest_xrays/'+name+'.jpg',quality=90)

os.mkdir('/content/final')
os.mkdir('/content/final/images')
for name in df['image_id'].values:
    shutil.move('/content/chest_xrays/'+name+'.jpg','/content/final/images')
mm=df.groupby('human_id')

#function to create a new dataframe
def two_images():
    im1=[]
    im2=[]
    rep=[]
    idsu=[]
    for i in df['human_id'].unique():
        idsu.append(i)
        #print(i)
        s=mm.get_group('{}'.format(i))
        #print(s)
        #print(len(s))
        if len(s)==1:
            im1.append(s['image_id'].values[0])
            im2.append(s['image_id'].values[0])
            rep.append(s['reports'].values[0])
        elif len(s)!=1:
            im1.append(s['image_id'].iloc[0])
            #print(im1)
            im2.append(s['image_id'].iloc[1])
            #print(im2)
            rep.append(s['reports'].iloc[0])
            #print(rep)
    dataf=pd.DataFrame()
    dataf['patient_id']=idsu
    dataf['image1_id']=im1
    dataf['image2_id']=im2
    dataf['report_joint']=rep
    return dataf

final_df=two_images()

#cleaning text
agx=[]
for i in range(3338):
    #remove punctuations and convert to lower case
    agx.append(''.join([word.lower() for word in final_df["report_joint"][i] if word not in string.punctuation]))
k=[]
for i in range(len(agx)):
    k.append(''.join(agx[i]))
final_df['report']=k

#remove 'xxx' characters
new=[]
for i in final_df['report']:
    tmp=re.sub(r'xxx*','',i)
    new.append(re.sub(r'/d','',tmp))

final_df['report']=new
final_df.drop('report_joint',inplace=True,axis='columns')
final_df.head()
final_df.to_csv('finalized_df.csv')

#train and test
final_df_copy=final_df.copy()
train=final_df_copy.sample(frac=0.8,random_state=0)
test=final_df_copy.drop(train.index)

#add start and end of sequence strings
def seq(reports):
    reports = 'startseq' + ' ' + reports + ' ' + 'endseq'
    return reports
train['report']=train['report'].apply(lambda x:seq(x))
test['report']=test['report'].apply(lambda x:seq(x))
