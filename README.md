
# Music Genre Predictor
## Paper implemented: 

J. Mehta, D. Gandhi, G. Thakur and P. Kanani, "Music Genre Classification using Transfer Learning on log-based MEL Spectrogram," 2021 5th International Conference on Computing Methodologies and Communication (ICCMC), 2021, pp. 1101-1107, doi: 10.1109/ICCMC51019.2021.9418035.


## About the paper

## Literature Review

In the implemented paper, under section 2: Related Work, section A: ‘Literature review’, the authors review and reference a few other papers written by other authors. They discuss the improvement that each paper offers with respect to the original problem, which is classifying music based on genre.

The following is the section from the paper.

The authors of [7] start with defining the problem and how a hierarchy of musical genres is important and also explore the importance of classifying audio files into these genres. After this, the authors define three feature sets namely timbral texture, rhythmic content and pitch content for the proposed classification task at hand. After this, the authors of [7] use various statistical pattern recognition classifiers such as in order to classify full length as well as frame-based audio files.

The authors in [8] propose an automated classification system for music genres. For the same, they first extract features such as MFCC vector, spectral centroid, chroma frequencies, zero-crossing rate and spectral roll-off from the audio files. The authors then use various classifiers into an ensemble of various combinations such as AdaBFFs, SRCAM and MAPsCAT in order to classify the audio files from the GTZAN dataset.

The authors in [9] deal with the problem of automatic classification of music by artist identification. The authors use only audio level features along with SVMs for the same. The artist identification is done on a dataset of 1200 pop songs distributed among 18 different artists. The authors also postulate that training classifiers and testing them on same albums causes the KL divergence to surpass the Mahalanobis distance between MFCC statistics vector.


The authors in [10] propose an algorithm to use AdaBoost in order to select specific features which have been extracted from an audio file and then aggregated. The paper mainly deals with the postulation that classifying aggregated features from parts of audio leads to better results than doing the same over entire audio files. The system proposed in the paper was claimed to be the second best system for music genre classification at MIREX 2005.

The authors introduce a new feature extraction method, called the DWCHs in [11]. Explaining the basics of how data could be gathered and interpreted from sounds, they compare different models for Content Based Music Classification. The comparison is done between 4 algorithms - Support Vector Machine (SVM)s, K-Nearest Neighbors (KNNs), Gaussian Mixture Models (GMMs) and Linear Discriminant Analysis (LDA). Training on a dataset of about 1000 songs, having 100 of each genre, the experimentation results showed that SVMs were the best classifiers for Content Based Music Classification.

In [12], the authors approach Music genre classification by using spectrograms. Through the time-frequency images generated, texture features are extracted. The experimentation included local feature extraction from 2 datasets, i.e. the Latin Music Database and the ISMIR 2004 dataset. The results were obtained with and without local feature extraction. The authors succeeded in obtaining accuracy higher than the best results obtained in the MIREX 2010 competition for the LMD, however, the accuracy was not the best in the latter dataset.

The authors of [13] propose a rather novel approach for music genre classification using a mix of different sets of features. This combination is finally used as an ensemble which shows great accuracy. Using three databases, namely the LMD, the ISMIR 2004 database and the GTZAN genre collection, the experimentation comprises of obtaining timbre features from audio signals, calculating statistical measures such as the texture window and modulation spectrum, performing a feature selection to increase the performance and finally training an ensemble on the final data. The method proposed by the authors outperforms several other published procedures.

Temporal feature integration, a method proposed by [14] is one that combines all feature vectors into one, in order to get the necessary temporal information out of the data. The paper proposes a multivariate autoregressive feature model for music genre classification. Using 2 datasets of 5 and 11 genres, 4 different classification schemes are employed for the model training, and their performance was compared to that of a human. The paper claims that its proposed MAR features perform better than the other features, but on the other hand increase the computational complexity.



Under section B: Research Gap, the authors talk about how the previously cited papers all have a feature-based approach to the problem and how their project focuses more on a non-feature based approach by converting the original audio files into distinct spectrograms. This approach has the advantage of not needing to extract various MFCC features, such as timbral texture and spectral roll-off, from the audio file. In a spectrogram approach, these features are not necessary as the image is passed through a CNN and the various dependencies of the audio file are learnt by the parameters of the convolutional layers.
 The authors also point out that although spectrogram approaches are being used in some models, transfer learning is not as wide-spread and as such they will be using a transfer learning approach. Transfer learning helps to speed up the learning process as the core features have already been learnt by the model and therefore better classification can take place.
This is the reason the authors use various pre-trained models for fine tuning.



## Data preparation

Primarily the GTZAN dataset consists of audio files belonging to 10 genres namely classical, blues, country, disco, hip-hop, jazz, metal, pop, reggae and rock. A derivative of this GTZAN dataset was used as GTZAN is well known in the Music Information Retrieval Community as a baseline dataset for genre classification projects.
Audio files were taken from the GTZAN dataset and converted into spectrograms using a sampling rate of 44100 Hz which is considered a standard sampling rate for empirical practices.

The spectrogram images are then resized to a size of 224x224 for uniformity while working with pre-trained ImageNet models, after which some data augmentation techniques are applied.
Data augmentation is a regularization technique which is used to perform some random operations on the image before feeding it to the learner, which would then result in an increased variation of the data being used for learning and help the model to generalize better for inference. 
Since spectrograms are sensitive to the orientation of the image, we could not use the usual methods of rotation and cropping. We instead did augmentation on the audio samples directly, by adding in white noise, as well as cutting out segments of the samples and generating new spectrograms from them.
The spectrograms were also normalized as described in [https://web.archive.org/web/20220108092102/http://enzokro.dev/spectrogram_normalizations/2020/09/10/Normalizing-spectrograms-for-deep-learning.html]




## Methodology

## Transfer learning

Trained or ‘learned’ models can be saved for future use. The configurations of the various layers of the network can be remembered and loaded into a new network with similar layer dimensions. In this case knowledge of the learned model is ‘transferred’ to the new untrained model. This type of deep learning is known as transfer learning.
Transfer learning is very efficient in terms of training time and computational power requirement. During the application of transfer learning, the last few layers of the pre-trained network are taken away and new layers specific to the model take their place. This speeds up the training process and makes the performance of the deep learning model better.

## Improvement

## Mixup Training

Mixup is one of the augmentation methods which allows us to prevent overfitting of datasets with huge parameters. Mixup generates additional samples during training by linear interpolation of various samples and their labels.Different linear interpolation will produce different new samples, which makes the neural network have more opportunities to sample new data to avoid overﬁtting
 
## Test time augmentation
 
The Test Time Augmentation performs random modifications to the test samples. Instead of showing the regular sample, only once to the trained model, we will show it the augmented samples several times. We will then average the predictions of each corresponding image and take that as our final guess. By averaging our predictions, on randomly modified images, we are also averaging the errors.
 
 
## Architectures

The dataset was trained on ResNet34, ResNet50, VGG16, and AlexNet. The accuracies obtained from training on these models were then compared to have an understanding of which model was most suitable for the problem of classification of music genres.

AlexNet: AlexNet is the name of a convolutional neural network (CNN) architecture, designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton, who was Krizhevsky's Ph.D. advisor.
AlexNet competed in the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. The original paper's primary result was that the depth of the model was essential for its high performance, which was computationally expensive, but made feasible due to the utilization of graphics processing units (GPUs) during training.
AlexNet contained eight layers; the first five were convolutional layers, some of them followed by max-pooling layers, and the last three were fully connected layers. It used the non-saturating ReLU activation function, which showed improved training performance over tanh and sigmoid.


VGG: Given below is the architecture of VGG.
Input. VGG takes in a 224x224 pixel RGB image. For the ImageNet competition, the authors cropped out the center 224x224 patch in each image to keep the input image size consistent.
Convolutional Layers. The convolutional layers in VGG use a very small receptive field (3x3, the smallest possible size that still captures left/right and up/down). There are also 1x1 convolution filters which act as a linear transformation of the input, which is followed by a ReLU unit. The convolution stride is fixed to 1 pixel so that the spatial resolution is preserved after convolution.
Fully-Connected Layers. VGG has three fully-connected layers: the first two have 4096 channels each and the third has 1000 channels, 1 for each class.
Hidden Layers. All of VGG’s hidden layers use ReLU (a huge innovation from AlexNet that cut training time). VGG does not generally use Local Response Normalization (LRN), as LRN increases memory consumption and training time with no particular increase in accuracy.
Due to this proposed architecture VGG achieved state of the art results in ImageNet Large-Scale Visual Recognition Challenge and as such was used to by the authors for fine-tuning on the dataset and comparing accuracies.

ResNet(Residual Neural Networks) : training becomes difficult when the model becomes deeper as more and more layers are stacked due to problems such as vanishing gradients and limits of computational complexity. Resnets solve this problem by introducing skip connections and skip blocks. Skip connections connect activation layers to further layers by skipping some layers in between which form a residual block. ResNets are made by stacking these residual blocks.
The authors have used two variations of this, namely, ResNet34 and ResNet50.

Below are the results of our implementation of the paper.


**AlexNet

![image](https://user-images.githubusercontent.com/7345512/211477717-9bd922d6-2bd4-4a0e-8d45-0ab2cadea3a7.png) ![image](https://user-images.githubusercontent.com/7345512/211477579-5f60ed5c-0fef-43d2-9135-67c5ff1cc90d.png) ![image](https://user-images.githubusercontent.com/7345512/211477682-4d36cc4b-6d7e-4370-bfb7-62dd527ebe41.png)



Training Statistics				Validation and Training Loss Curve       

Peak accuracy is 79.3970%

**VGG

![image](https://user-images.githubusercontent.com/7345512/211477792-d87fb760-5515-4bc1-96ab-f6e0fb44cdbe.png) ![image](https://user-images.githubusercontent.com/7345512/211477748-8b1ebffb-84ce-4d6f-a0f8-f4f96f334645.png) ![image](https://user-images.githubusercontent.com/7345512/211477767-decf7eb4-c14f-4d24-91bf-853d623d472c.png)   


Training Statistics				Validation and Training Loss Curve       

Peak accuracy is 87.4372%

**ResNet50

![image](https://user-images.githubusercontent.com/7345512/211477877-e129b3a0-5836-49f3-91e1-1ce3e1e49090.png) ![image](https://user-images.githubusercontent.com/7345512/211477818-2fbb0e46-d2e8-4c28-9125-1ef48335509d.png)   ![image](https://user-images.githubusercontent.com/7345512/211477842-e814abdd-7003-4c45-8edf-a4e2bd4c1e26.png)


Training Statistics				Validation and Training Loss Curve       

Peak accuracy is 88.4422%









**ResNet34

![image](https://user-images.githubusercontent.com/7345512/211477942-3bd3a844-7e77-4c33-b421-1980c36cd6e1.png)![image](https://user-images.githubusercontent.com/7345512/211477986-f6b86e1a-537e-4daf-be63-3528e24ce295.png) ![image](https://user-images.githubusercontent.com/7345512/211477899-21df812e-858a-4469-95a7-ccdd9522f1bc.png)   



Training Statistics				Validation and Training Loss Curve       

Peak accuracy is 85.9297%




The above images represent the results of the reviewed paper without any of the optimisations we have implemented. Each of the 4 models were trained for 10 epochs each from their pretrained weights with a base learning rate of 0.01. The results follow the same trend as shown in the paper.
AlexNet, being the oldest model and with not that many layers, is not able to learn that well in only 10 epochs, with a peak accuracy of 78.8%. VGG, on the other hand is a much newer and deeper model, and hence is much more accurate with a lower  number of training epochs, but with a much higher time taken per epoch.
For both Resnet-50 and Resnet-34, the benefits of skip connections show itself, as it has similar accuracies to vgg-16, but with a lesser training time.

For all the above models (especially the heavier ones), we can see that they do not generalize very well as shown by the loss graphs for both validation and training losses not going down evenly.

To improve generalization as well as increase accuracy, we used ResNet rs50(refer to paper here) with mixup, tta as well as data augmentation and the new results are shown below.



**Resnet rs50


![image](https://user-images.githubusercontent.com/7345512/211480233-001c796f-ee34-4197-9132-9639174dee78.png) ![image](https://user-images.githubusercontent.com/7345512/211480275-2757acd0-e743-4b54-909d-69403b5a1cbd.png)




Training Statistics				Validation and Training Loss Curve       

Peak accuracy is 90%



![image](https://user-images.githubusercontent.com/7345512/211480321-2a2f6896-0146-429a-81fd-7bee8be6e399.png) ![image](https://user-images.githubusercontent.com/7345512/211480375-e11a625e-c21f-49b3-807e-9283de18c755.png)



Training Statistics				Validation and Training Loss Curve       

Peak accuracy is 91.5% (after more training)



Mixup helps our model generalize much better, as shown by the loss curves over the 2 sets of epochs. The validation losses are much lower than the previous results, and the training losses are still not very low, which indicates that there is no overfitting. It is possible that we will get even better results if we were to train our model for even more epochs(or use a deeper model), but that gain would not have been as significant compared to the computation cost.

![image](https://user-images.githubusercontent.com/7345512/211480423-7ac5a0e7-2167-4b67-8e92-b43a3890523d.png)

Test time augmentation

After implementing test time augmentation, accuracy is similar to without tta, but since the model is inferring on various augmented versions of the input (we chose 8 such augmentations), the model makes a much better inference. 




## Conclusion

In totality, this report contains our findings on training the GTZAN dataset on five distinct architectures, namely, AlexNet, ResNet34, ResNet50, VGG and ResNet rs50.

We observe that ResNet rs50 gives us our highest accuracy of over 91% when trained over several epochs, followed by ResNet50, VGG, ResNet34, and finally AlexNet.

Unsurprisingly, AlexNet gives us the lowest accuracy over ten epochs, being a relatively antiquated model compared to the other architectures used. We have also observed empirically that the skip connections of both ResNet architectures are beneficial in improving accuracy, coupled with a short training time.

The VGG architecture, although roughly equal to ResNet in terms of accuracy, has been seen to have a much higher training time per epoch, which particularly lends itself to its usefulness in fine-tuning.

Ultimately, introducing mix-ups further increases the performance of our model, up till the point where prolonging the training process yields us diminishing returns. 

 
Names and ID numbers of the group Members:
 
- Anubhab Khanra 2020A7PS2144H
- Neeraj Gunda 2020A7PS0169H
- Sivaram Padmasola 2020AAPS0387H
- Madhav Srinivas Nathavajhula 2020A3PS0654H
 

 


