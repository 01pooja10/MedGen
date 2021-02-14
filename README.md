# MedGen

This project is a TensorFlow reimplementation of the paper __On the Automatic Generation of Medical Imaging Reports__ by Jing et all, published in the year 2018.

Check out the paper [here!](https://arxiv.org/abs/1711.08195)

## Paper Inspiration
Medical images like xrays, CTs, MRIs and other type of scans are used for diagnosis of a lot of diseases. Specialized medical professionals read and interpret these medical images. Report writing for these scans can be time-consuming, and to address this issue, we looked into automatic generation of these reports. 

## Sample Medical Report
A medical report has three main points:
- Impressions, which provide diagnosis
- Findings, which lists all observations
- Tags, which list keywords which represent the critical information in the findings

<p align="center">
<img src="https://github.com/01pooja10/Medical-Report-Generator/blob/main/misc/sample_report.png" alt="Sample Report">

## Dataset
The dataset used in the paper was the Indiana University Chest X-Ray Collection [(IU X-Ray)](https://www.kaggle.com/raddar/chest-xrays-indiana-university) (Demner-Fushman et al., 2015), which is a set of chest x-ray images paired with their corresponding diagnostic reports.
The images were obtained from [here](https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d) and the reports were obtained from [here](https://academictorrents.com/details/66450ba52ba3f83fbf82ef9c91f2bde0e845aba9).

Due to computational difficulties, we used a sample set of 1000 scans for training and 200 scans for testing, the details of which are present in the directory /data.

## Model Components
The architecture proposed by the paper is shown below. 

<p align="center">
<img src="https://github.com/01pooja10/Medical-Report-Generator/blob/main/misc/model_structure.png" alt="Proposed Architecture">


The three main proposals of the paper are:
- A multi-task framework which jointly performs the prediction of tags and generation of paragraphs for reports
- Co-attention mechanism which takes visual features as well as semantic features into account
- Hierarchical LSTM model

## Implementation details
 - The zip files containing images and reports were mounted from Google Drive.
 - A dataset was prepared through a data cleaning process that consists of two images per report, one frontal and one lateral view.
 - Reports were extraced from .xml files and the frontal and lateral views were combined to prepare the above mentioned dataset and this was used to generate features.
 - glove.840B.300d was used for obtaining vector representations and generating the embedding matrix. It is available [here](https://nlp.stanford.edu/projects/glove/).
   - To run the model, download the glove file and add to MedGen folder.
 - Features were extracted using DenseNet121 model loaded with ChexNet weights (available [here](https://www.kaggle.com/theewok/chexnet-keras-weights)). The paper used a VGG-19 network.
   - The features are available in ./features directory. 
 - The features were fed into a model with the following structure
<p align="center">
<img src="https://github.com/01pooja10/Medical-Report-Generator/blob/main/misc/attn_mod.jpg" height="400" alt="Model structure">
 
 - To train the model, run *encoder_decoder.ipynb* in root directory.
 
 ## Generated Report
- The model was trained for 10 epochs. Due to computational difficulties, we were unable to train for more epochs and hence the model did not converge. 
- Final BLEU score was 0.643
 <p align="center">
<img src="https://github.com/01pooja10/Medical-Report-Generator/blob/main/misc/generated_report.png" height="300" alt="Generated Report">

## Dependencies
- TensorFlow 2.4.1
- Keras 2.4.3
- Numpy
- Pandas 1.1.5
- Sklearn 0.23.2
- PIL 8.0.1
- Nltk 3.5
- Matplotlib 3.3.2
- Opencv 4.5.1
- Tqdm 4.50.2
- OS

## To-do
- [ ] Complete tag prediction using MLC
- [ ] Integrate semantic features in co-attention model

## Contributors 
- [Indira Dutta](https://github.com/indiradutta)
- [Pooja Ravi](https://github.com/01pooja10)
- [Sashrika Surya](https://github.com/sashrika15)

For any queries, please open an issue at the repository, or email any of the contributors.

