
# Matroid Challenge
Author: Gordon Yeh
Email: gordonforjob@gmail.com


## Platform
Google Colab

## Data Collection and Prepare
1. Download pictures data from 
https://s3.amazonaws.com/matroid-web/datasets/agegender_cleaned.tar.gz
2. Load all Images from folder /combined/aligned using ImageDataGenerator(), Split 80% of whole dataset as training, 10% as validation, and 10% as test

## Model Building 
1. Researched on model transformation, tried implementing ONNX, and Caffe-tensorflow, but did not work due to many Python library version issues
2. Alternatively built VGG architecture from scratch using keras, and loaded matconvnet pre-trained weights into it

## Training and Fine-tuning
1. For transfer learning on image gender classification, Froze all convolutional layers for the first half of model
2. Left fully-connected layers architecture remain same, and change final activation function to "Sigmoid", because it is binary. Loss function chose binary cross entropy.
3. Start to train on part of dataset, but could not converge.
4. Modified fully-connected nodes amount from 4096 to 512; Removed the Relu layer in output layers; Added Batch Normalization layer 
5. Model started to converge, trained three epochs (due to time limits) with batch size 64 using optimizer "Adam", and chose model with highest validation accuracy 
6. To reproduce whole process, run all cells in .iPython file will automatically finish implementation
7. Loss and accuracy of training and validation

|        |TRAIN LOSS|VALID LOSS|TRAIN ACC|VALID ACC| 
|--------|-----------|-----------|----------|---------|
| Epoch1 | 0.3415     | 0.3143     | 0.9411    | 0.9375 |  
| Epoch2 | 0.2549     | 0.2350     | 0.9664    | 0.9625 | 
| Epoch3 | 0.1996     | 0.1724     | 0.9767    | 0.9625 |

## Result and Metrics
1. Overall Test Accuracy: 0.965
2. Confusion Matrix

|        |Predicted_Female|Predicted_Male|
|--------|-----------|-----------|
| True_Female | 1388   |   43   |
| True_Male |  62   | 1450     | 

3. F1-Score

 |        |Precision|Recall|F1-Score|
|--------|-----------|-----------|--|
| Female | 0.96   |   0.97   | 0.96
| Male |  0.97   | 0.96     | 0.97

## Reference

 - ImageDataGenerator Implementation https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
 - Convert Matlab model to Tensorflow https://sefiks.com/2019/07/15/how-to-convert-matlab-models-to-keras/


