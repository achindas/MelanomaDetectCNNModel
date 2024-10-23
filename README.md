# Melanoma Detection through Skin Image Analysis - CNN Deep Learning Model

## Table of Contents
* [CNN Overview](#CNN-overview)
* [Problem Statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Approach for Modeling](#approach-for-modeling)
* [Classification Outcome](#classification-outcome)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)

## CNN Overview

### What is a Convolutional Neural Network (CNN)?
A Convolutional Neural Network (CNN) is a specialized deep learning model designed for processing grid-like data structures, such as images. Unlike traditional neural networks, CNNs leverage spatial relationships between pixels to extract important features (edges, textures, patterns) automatically without requiring extensive manual feature engineering.

### Where Are CNNs Used?
CNNs are predominantly used in the field of computer vision but are also applied in other domains that require pattern recognition, such as:

* **Image Classification:** Identifying objects in an image (e.g., cats vs. dogs).
* **Object Detection:** Locating specific objects within an image.
* **Image Segmentation:** Dividing an image into meaningful segments.
* **Facial Recognition:** Detecting and recognizing human faces.
* **Medical Image Analysis:** Detecting diseases from MRI or CT scan images.
* **Natural Language Processing (NLP):** Though less common, CNNs have also been used for text classification tasks.

### Architecture of CNN
The architecture of CNNs typically consists of the following layers:

**Input Layer:**

The input layer receives raw data, typically images. Images are represented as multi-dimensional arrays, such as 2D arrays for grayscale images and 3D arrays for colored images (RGB channels).

**Convolutional Layer:**

The convolutional layer is the core building block of CNNs. It applies filters (kernels) to the input image and performs convolution operations to extract feature maps. Each filter is responsible for detecting specific patterns like edges or textures.

**Activation Function (ReLU):**

After applying the convolution operation, an activation function like ReLU (Rectified Linear Unit) is used to introduce non-linearity. This ensures the network can learn complex patterns.

**Pooling Layer (Subsampling):**

Pooling layers are used to reduce the spatial dimensions (width, height) of the feature maps. It helps to reduce computational load, control overfitting, and retain essential features.

**Fully Connected Layer (Dense Layer):**

Once feature extraction is complete, the fully connected layer is used to classify the image based on the learned features. The output of the convolutional and pooling layers is flattened into a vector and passed to this layer.

**Output Layer:**

The final layer of the network provides the prediction. For classification tasks, a softmax activation function is commonly used to produce a probability distribution across multiple classes.

### CNN Hyperparameters
Several important hyperparameters govern the performance of CNNs:

**Filter (Kernel) Size:**

Specifies the size of the filter used in the convolutional layers. Common choices are 3x3 or 5x5. Larger filters capture more complex features but require more computation.

**Stride:**

Determines how much the filter moves over the input matrix. 

**Padding:**

Controls how the borders of an image are handled during convolution. 

**Learning Rate:**

Controls the step size at each iteration while moving toward the minimum of the loss function.

**Batch Size:**

Determines the number of training samples to pass through the model before updating the weights.

**Number of Epochs:**

Defines the number of times the model processes the entire training dataset. More epochs allow the model to learn better but also increase the risk of overfitting if not regularized properly.

**Dropout Rate:**

Dropout is a regularization technique used to prevent overfitting by randomly dropping out neurons during training.

**Optimizer:**

Optimizers like Adam, SGD, and RMSProp are used to update the weights of the model during backpropagation. Adam is widely used due to its adaptive learning rate.

**Regularization (L2, L1):**

Regularization techniques such as L2 or L1 regularization are used to penalize large weights and reduce the risk of overfitting.

### Other Useful Information about RNN

**Data Augmentation:**

To prevent overfitting, CNN models often use data augmentation, which artificially increases the size of the dataset by applying transformations such as flipping, cropping, rotation, and zooming to the input images.

**Transfer Learning:**

CNNs often benefit from transfer learning, where a pre-trained model (like VGG, ResNet, or Inception) is fine-tuned for a specific task. This is particularly helpful when the available dataset is small.

**Challenges:**

CNNs are computationally expensive and require large amounts of data to train effectively. However, they excel in feature detection and classification tasks once trained, even outperforming traditional machine learning methods in many areas.


**Convolutional Neuwal Network (CNN)** has been utilized in this case study in a step-by-step manner to understand, analyse, transform and model the image data provided for the analysis. The approach described here represent the practical process utilised in industry to predict categorical target parameters for business.


## Problem Statement

To build a **CNN based model** which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

The dataset for this exercise consists of images of malignant and benign oncological diseases, which were formed from the **International Skin Imaging Collaboration (ISIC)**. All images were sorted according to the classification taken with ISIC.

The data set contains the following diseases:

* Actinic keratosis
* Basal cell carcinoma
* Dermatofibroma
* Melanoma
* Nevus
* Pigmented benign keratosis
* Seborrheic keratosis
* Squamous cell carcinoma
* Vascular lesion

## Technologies Used

Python Notebook in Google Collab environment with GPU support has been used for the exercise. Apart from ususal Python libraries like  Numpy, Pandas and Matplotlib, there are Deep Learning specific libraries used to prepare, analyse, build model and visualise data. The following model specific libraries have been used in the case study:

- tensorflow
- keras
- skimage
- Augmentor


## Approach for Modeling

The following steps are followed to build the Naive Bayes model for IMDb sentiment analysis:

1. Import & Understand Data
2. Create Datasets
3. Visualise Dataset
4. Preprocess, Build & Train Model
5. Augment Dataset & Remodel
6. Handle Class Imbalance & Remodel
7. Make Predictions
8. Conclusion

Some distinguishing processes in this approach include,

- Building the Training and Validation datasets by reading image files from disk using `keras.preprocessing.image_dataset_from_directory` function

- Creation of multi-layer CNN model using `Sequential` function of `tensorflow` library having required number of convolutional, dropout, flatten and dense layers

- Compilation of the CNN model using `adam` optimiser and `SparseCategoricalCrossentropy` loss functions

- Implementation of data augmentation using the keras preprocessing transformation layers e.g. `tf.keras.layers.RandomFlip`

- Handling of class imbalance by creating new images from existing ones using `Augmentor` package


## Classification Outcome

A CNN model is built using `keras` with three convolutional layers and one dense intermediate layer to analyse the images of various cancer types to particularly detect melanoma among patients.

The basic model with **less number of images had overfitting issue** manifested through a significant gap between training and validation accuracies.

Subsequently **data aumentation technique** was used to increase the number of images though random transformation. This helped in addressing the overfitting issue, but it **brought down the accuracy** to near 55%.

Later the class-wise **data imbalance issue** was addressed by augmenting each class with additional 500 images. This helped improving the accuracies above 80% while keeping the overfitting issue in check.

So we have a very good CNN model at the end which is able to accurately predict various cancer types including melanoma based on the images.


## Conclusion

Convolutional Neural Networks are powerful tools for various tasks related to image and pattern recognition. By applying layers of convolutions, pooling, and fully connected layers, CNNs can automatically learn features from data, making them highly efficient for tasks that require feature extraction and classification.


## Acknowledgements

This case study has been developed as part of Post Graduate Diploma Program on Machine Learning and AI, offered jointly by Indian Institute of Information Technology, Bangalore (IIIT-B) and upGrad.