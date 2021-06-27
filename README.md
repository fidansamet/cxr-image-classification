# CXR Image Classification

This repository contains the implementation of nearest neighbor and weighted nearest neighbor algorithms to detect Covid-19 disease from images. By extending these algorithms, it classifies CXR images as Covid-19 positive case, viral pneumonia or normal CXR.

## Run

### Train

    python main.py

### Test

    python main.py --phase test

For more options, please refer to options.py


## Dataset

Used dataset contains 1200 Covid-19 positive case, 1345 Viral Pneumonia and 1341 Normal CXR images in total. Below is the table of image numbers for train and validation sets. The remainder of the dataset is reserved as test set.

| Type of CXR | # Images |
| :- | -: |
| Covid-19 Positive Case | 960 |
| Viral Pneumonia | 1073 |
| Normal | 1076 |



## Image Features

In order to classify images, features are needed to be extracted from them. Thus, by obtaining the information represented by the images, we can perform the classification task by calculating the similarity between the images. To obtain the information from images, we can convert the image from matrix to feature vector or we can use already existing methods. The image features I use are as follows:

**1. 64x64 Image Feature:** I resize the images to 64x64. Then I reduce the number of channels from 3 to 1 by converting the black and white CXR images from red-green-blue to gray-scale. Then I flatten the obtained one-dimensional matrix and transform it into a feature vector. Lastly, I normalize this vector if specified. Therefore I obtain the features of the gray-scale image resized to 64x64. Since 64x64 feature vector is not so small, it contains detailed information of image.

**2. Tiny Image Feature:** I resize the images to 16x16. Then I reduce the number of channels from 3 to 1 by converting the black and white CXR images from red-green-blue to gray-scale. Then I flatten the obtained one-dimensional matrix and transform it into a feature vector. Lastly, I normalize this vector if specified. Therefore I obtain the features of the gray-scale image resized to 16x16. Since 16x16 feature vector is really small, it contains smattering information of image.

**3. Shape Features:** Shape features are useful in image classification. For instance, detecting edges gives a broad information about image. I use Canny edge detection to obtain the histogram of detected edges in images. After finding the edges in the images, I resize the edge matrix to 64x64 due to computation power and timing issues. Then I flatten the obtained one-dimensional matrix and transform it into a feature vector. Lastly, I normalize this vector if specified. Therefore I obtain the features of edges which contain the information about shapes in images.

**4. Texture Features:** Texture features are also useful in image classification. For instance, applying texture filters helps us to extract texture information from images. I use Gabor filter to obtain the feature extractions in images. After obtaining the texture information in the images, I resize the texture matrix to 64x64 due to computation power and timing issues. Then I flatten the obtained one-dimensional matrix and transform it into a feature vector. Lastly, I normalize this vector if specified. Therefore I obtain the Gabor filtered features which contain the information about textures in images.

**5. Histogram of Oriented Gradients (HOG):** Histogram of oriented gradients is a feature descriptor mostly used for object detection. With feature descriptors, we can obtain the simplified but important representations of the images. Therefore I think this feature that I worked on before is suitable for our image classification task. After getting the feature descriptors, I resize the texture matrix to 64x64 due to computation power and timing issues. Then I flatten the obtained one-dimensional matrix and transform it into a feature vector. Lastly, I normalize this vector if specified. Thus I obtain the feature descriptors that give important information about images.

**6. VGG-19 Network Deep Image Features:** VGG-19 is a deep Convolutional Neural Networks (CNN) used for image classification task. Below is the architecture of this network. This network mainly classifies 1000 classes. Therefore the size of the last fully connected layer is 1000. Since I only want to extract deep image features from this network, I change model mode to evaluation and remove the last fully connected layer. So when I forward an image to this edited network, I obtain an image feature vector of size 4096. Note that to forward an image to this network, the image must be preprocessed and convert to tensor.

![](https://www.researchgate.net/profile/Clifford-Yang/publication/325137356/figure/fig2/AS:670371271413777@1536840374533/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means.jpg)

After preprocessing the images, I forward the images to last fully connected layer removed VGG-19 network and obtain the feature vectors. Lastly, I normalize this vector if specified. Therefore I obtain the extracted deep image features which give information about images.

In some experiments, instead of working with a single feature, I obtained features by concatenating different features. Since combining the features increases the size of the feature vector and increases the computation time, I could not try every combination. In the future, keypoint extractors like Scale-Invariant Feature Transform (SIFT) and Speeded Up Robust Features (SURF) can also be experimented.



## k-Fold Cross Validation

k-fold cross validation is an useful technique to tune the hyperparameters. In this technique, the dataset is randomly divided into k partitions. These partitions are iterated on and the current partition is separated as the validation-set, while the remaining partitions form the train-set. The validation score is obtained by using the separated partition at the end of each iteration. 

The appropriate number of folds depends on the size of the dataset. Using large k values provide less bias towards overestimating but computation time becomes high. Therefore I use 3, 4, 5 and 6 folds of shuffled data to validate the models properly. After shuffling the training data, I split it into k partitions. While iterating on folds, I reserve the current fold for validation and use the rest as a train-set. I calculate the validation accuracy with the partition separated as a validation-set for each fold.



## k-Nearest Neighbors

The nearest neighbor algorithm is a simple classification algorithm. It memorizes all the train-set samples and their labels. When a test sample is given, it predicts the label of that sample by finding the most similar train-set sample(s). When more than one nearest neighbor is considered, the majority label is predicted. This approach can be extended to weighted nearest neighbor algorithm by taking the distances into account so that the closest sample gets higher vote.

In the implementation of k-nearest neighbor algorithm, I store the train-set samples and their labels at first. When a test sample is given, I calculate the distances between all the train-set samples and given test sample. I consider different distance measures in this step which are Euclidean, Manhattan, Hamming and  Minkowski (p=3). After obtaining distances, I sort them to get smallest ones. After getting the closest neighbors according to the number of neighbors needed to consider, I obtain their labels. In k-nn prediction, I predict the most frequent label in those k-nearest neighbor labels as test sample label. In weighted k-nn prediction, I calculate the weights of k-nearest neighbors by taking inverse of distances and normalizing them. Then I count the votes of labels by summing the weights of the same classes. Finally I predict the top rated label as test sample label. Note that in case of equality in majority, I predict the nearest neighbor class among them. I repeat all these steps while experimenting for neighbor numbers from 1 to 10.



## Accuracy

The mean accuracy needed to be reported for each setup to measure the succes of classification method setups. Accuracy is calculated by dividing number of correctly classified samples to total number of samples and multiplying with 100. After calculating accuracies of each fold, I obtain the mean accuracy by taking the average of accuracies for each fold.



## Experimental Results

In the experiments, I tune hyperparameters such as k number and distance measure of k-nearest neighbor algorithm for each feature combination. Below is the test accuracies table of the best models of experiments in k-nearest neighbor algorithm.

| Feature | Test Accuracy (%) |
| :- | -: |
| 64x64 | 93.05 |
| 16x16 | 92.15 |
| Canny | 49.16 |
| Gabor | 80.31 |
| HOG | 91.25 |
| VGG-19 | 91.38 |
| Canny & Gabor (Norm) | 74.26 |
| 16x16 & HOG | 89.45 |
| 16x16 & HOG (Norm) | 91.76 |
| 16x16 & VGG-19 | 92.54 |
| 16x16 & VGG-19 (Norm) | 91.63 |
| HOG & VGG-19 | **94.21** |
| HOG & VGG-19 (Norm) | 92.41 |
| HOG & VGG-19 & 16x16 (Norm) | 92.28 |


Below is the test accuracies table of the best models of experiments in weighted k-nearest neighbor algorithm.

| Feature | Test Accuracy (%) |
| :- | -: |
| 64x64 | 92.41 |
| 16x16 | 92.28 |
| Canny | 52.38 |
| Gabor | 80.57 |
| HOG | 90.60 |
| VGG-19 | 90.86 |
| Canny & Gabor (Norm) | 72.46 |
| 16x16 & HOG | 88.42 |
| 16x16 & HOG (Norm) | 92.28 |
| 16x16 & VGG-19 | 92.92 |
| 16x16 & VGG-19 (Norm) | 91.76 |
| HOG & VGG-19 | **94.47** |
| HOG & VGG-19 (Norm) | 92.92 |
| HOG & VGG-19 & 16x16 (Norm) | 93.05 |
