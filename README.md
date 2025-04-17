# Image_processing_MobinetV2

Face Mask Detection Project using MobileNetV2¶
Author: Pham Ngoc Hieu + Nguyen Nhat Phap - HCMUTE

Description: This project implements a face mask detection model using MobileNetV2 with transfer learning.

The project includes data preprocessing, model training, evaluation and testing on new images.

* Importing Libraries and Packages¶
+ In this section, we import all the necessary libraries and packages required for the project.
+ These include libraries for deep learning (TensorFlow, Keras), data preprocessing, visualization, and image manipulation.

* Loading and Visualizing Dataset¶
+ In this section, we specify the paths to the dataset directories. The dataset consists of images of people wearing masks correctly, incorrectly, or without masks.
+ We then load a few example images from each category and visualize them.

Data Preprocessing and Augmentation¶
We use ImageDataGenerator to apply data augmentation techniques such as rescaling, zooming, and flipping.
This helps in generating more diverse training data and reducing overfitting.
We also split the dataset into training and validation sets, specifying the target size (224,224) for MobileNetV2.

Building the Model using MobileNetV2¶
We use the MobileNetV2 architecture, pre-trained on ImageNet, as the base model for transfer learning.
The model's top layers are replaced with custom layers suitable for our 3-class classification task (mask_weared_incorrect, with_mask, without_mask).

Adding Callbacks (Model Checkpointing and Early Stopping)¶
In this section, we implement ModelCheckpoint to save the best-performing model based on validation accuracy.
We also apply EarlyStopping to stop training if the validation performance degrades after a certain number of epochs.

Training the Model¶
We train the model for 20 epochs using the training data, with validation data being used to monitor performance.
ModelCheckpoint and EarlyStopping callbacks are used to optimize the training process.

Model Evaluation on Validation Data¶
After training, we load the best model and evaluate its performance on the validation set.
The accuracy of the model is printed as a percentage.

Detailed Evaluation using Confusion Matrix and Classification Report¶
In this section, we evaluate the model's performance using a confusion matrix and a classification report.
These metrics provide insights into the precision, recall, and F1-score of the model for each class.

Saving the Final Model¶
Save the final trained model to a file for future use or deployment.
