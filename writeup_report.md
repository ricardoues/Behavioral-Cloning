# **Behavioral Cloning Project** 

## Writeup 

---

**Behavioral Cloning Project**

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/image01.png "Center Lane Driving"
[image2]: ./images/image02.png "Left Lane Driving"
[image3]: ./images/image03.png "Right Lane Driving"
[image4]: ./images/image04.png "Distribution of steering angles"
[image5]: ./images/image05.png "Distribution of steering angles"
[image6]: ./images/image06.png "Right Lane Driving"
[image7]: ./images/image07.png "Flipped Right Lane Driving"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the simulator provided by Udacity and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
**Note**: Before to run the code you must install the carnd-term1 conda environment using the following instructions: 

[Configure and Manage Your Environment with Anaconda](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md)

Once you have done the above, you must run the following before using the simulator.

```sh
source activate carnd-term1
```


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline that I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the NVIDIA Architecture proposed in the videos. The network consists of a normalization layer (to avoid numerical optimization problems) followed by five convolutional layers (model.py lines 225-229), followed by four fully connected layers (model.py lines 231-237). In the first three convolutional layers, we use 5x5 filter sizes and  2x2 strides. Finally, we use 3x3 filter sizes and 1x1 strides in the last two convolutional layers. 

The data is normalized in the model using a Keras lambda layer (model.py line 222). Moreover cropping was used in order to provide meaningful information to the deep learning, for example, the sky is not relevant to the deep learning model (model.py line 223). 

#### 2. Attempts to reduce overfitting in the model

The deep learning model contains dropout layers in order to reduce overfitting (model.py lines 232-236). The model was trained and validated several times to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In order for the model to achieve satisfactory performance I had to use three laps. 


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to begin with simpler models and after that, use architectures more complex such as Lenet and NVIDIA architecture.

My first step was to use a convolution neural network model with two convolutional layers and three fully connected layers. I thought this model might be appropriate as a starting point since that always is a good practice to start with parsimonious models.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (this was done using the parameter validation_split from Keras). I found that my first model was underfitting that is why I use in the next step LeNet.

LeNet architecture shown the same problem, underfitting for this reason we have to use a more complex architecture such as NVIDIA architecture. To combat the overfitting, I modified the model, adding dropout in the fully connected layers. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I notice that the steering angles localized in the center are more common and that is the reason that the model cannot generalize well. We used oversampling to fix it.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a normalization layer, followed by five convolutional layers, followed by four fully connected layers. I used dropout as a way to regularize the model. 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center, left and right lane driving. Here are an examples of images of center, left and right lane driving:

![alt text][image1]

![alt text][image2]

![alt text][image3]


We examine the distribution of the steering angles by means of a histogram which is shown as follows: 

![alt text][image4]

The data is not uniform, that is why the model is specialized in the steering angle localized in the center. In order to fix it, we are going to generate data that will be uniformly distributed with respect to the steering angles, a function was developed (model.py lines 121-178). The resulting distribution of the steering angles once the function was run is shown as follows:

![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would improve the capacity of the generalization of the deep learning models, we must remember that deep learning models are prone to memorize the data (overfitting) that is why we need a lot of data to fit this kind of models. For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]


After the collection process, I had 42000 number of data points. I then preprocessed this data by using normalization and cropping on the information of the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set by means of the parameter validation_split of the method fit from Keras. Due to we observe how the model works in the simulator, we think it is not neccessary to use a test set. I used an adam optimizer so that manually training the learning rate was not necessary. Finally, we have to add a code in drive.py (line 67) to take account of the following: 

Training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.


