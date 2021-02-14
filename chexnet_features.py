import os
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense
from keras.models import Model
import pandas as pd

dirname = os.path.dirname(os.path.realpath('__file__'))


chest2=DenseNet121(include_top=False,weights=None,input_shape=(224,224,3))
last2=chest2.output
last2=Dense(14,activation='sigmoid')(last2)

mod2=Model(inputs=chest2.input,outputs=last2)
mod2.load_weights(dirname+'/features/chexnet_weights.h5')
mod2=Model(inputs=mod2.input,outputs=mod2.layers[-2].output)
#mod2.summary()

#image preprocessing function
def img_preprocess(img):
    img=img_to_array(img)
    img=preprocess_input(img)
    img=cv2.resize(img,(224,224))
    img=img/255.0
    img=np.expand_dims(img, axis=0)
    return img

#loading the extracted features into a dictionary after concatenation
def get_features(data):
    chex_features={}
    for i,j,k,l in data.values:

        img1=Image.open(dirname+'/data/images/'+j+'.jpg')
        img1=img_preprocess(img1)
        img1_feat=mod2.predict(img1)

        img2=Image.open(dirname+'/data/images/'+j+'.jpg')
        img2=img_preprocess(img2)
        img2_feat=mod2.predict(img2)

        mod_in=np.concatenate((img1_feat,img2_feat),axis=1)
        mod_in=tf.reshape(mod_in,(mod_in.shape[0],-1,mod_in.shape[-1]))

        chex_features[i]=mod_in

    return chex_features


train=pd.read_csv(dirname+'/data/traindata.csv')
test=pd.read_csv(dirname+'/data/testdata.csv')

train_features=get_features(train)
train_feat=train_features.values()
train_feat_list=list(train_feat)

test_features=get_features(test)
test_feat=test_features.values()
test_feat_list=list(test_feat)

final_features={}
final_features={**train_features,**test_features}
