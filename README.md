# AgroFresh
# AgroFresh – AI Food Freshness Detector

AgroFresh is an Artificial Intelligence and Computer Vision based project designed to identify the type and freshness condition of selected fruits from an image.

The current version uses a transfer learning approach with **MobileNetV2** to classify fruit images into six categories: fresh apple, rotten apple, fresh banana, rotten banana, fresh orange, and rotten orange.

The project is being developed as an undergraduate engineering project with the aim of addressing a common real-life problem: identifying whether food is fresh or rotten using an image.

---

## Problem Statement

Food wastage is a common problem in homes, shops, supermarkets, restaurants, and food supply chains. Determining whether a fruit is fresh or rotten is often based on visual inspection, which can be subjective and inconsistent.

AgroFresh attempts to provide an automated AI-based solution where a user can provide an image of a fruit and the system predicts:

- The type of fruit
- Whether the fruit is fresh or rotten
- The confidence of the AI prediction

---

## Proposed Solution

AgroFresh uses a deep learning image classification model to analyze fruit images.

The current workflow is:

```text
                 Input Fruit Image
                         |
                         v
                Image Preprocessing
                         |
                         v
                 MobileNetV2 Model
                         |
                         v
                 Feature Extraction
                         |
                         v
                  Classification
                         |
              +----------+----------+
              |                     |
              v                     v
          Fruit Type           Freshness
              |                     |
              +----------+----------+



Current Classification Classes

The current model recognizes six classes:

Class	Description
freshapples	Fresh Apple
freshbanana	Fresh Banana
freshoranges	Fresh Orange
rottenapples	Rotten Apple
rottenbanana	Rotten Banana
rottenoranges	Rotten Orange

The model therefore performs both fruit identification and freshness classification as a six-class classification problem.

AI Model

The current AgroFresh model uses MobileNetV2 with Transfer Learning.

MobileNetV2 is a lightweight convolutional neural network originally trained on the ImageNet dataset. Its pretrained layers are used for extracting visual features, while additional classification layers are trained using the fruit dataset.

Model Architecture
Input Image
     |
     v
Resize to 224 x 224
     |
     v
MobileNetV2
     |
     v
Global Average Pooling
     |
     v
Dense Layer (128 neurons)
     |
     v
Dropout (0.5)
     |
     v
Softmax Output
     |
     v
6 Fruit/Freshness Classes
Dataset

The model was trained using a Fresh and Rotten Fruits image dataset containing images of fresh and rotten fruits.

The dataset used for the current implementation contains:

Fresh Apples
Fresh Bananas
Fresh Oranges
Rotten Apples
Rotten Bananas
Rotten Oranges

The dataset is divided into training and testing images.

A validation subset is also created from the training data during model training.

Data Preprocessing

Before training, images are:

Resized to 224 × 224 pixels
Normalized to values between 0 and 1
Randomly rotated
Horizontally flipped
Shifted horizontally and vertically
Zoomed
Sheared

These augmentation techniques help the model learn from variations in the appearance of fruits.

Training

The model uses:

Model              : MobileNetV2
Learning Method    : Transfer Learning
Input Size         : 224 × 224 × 3
Batch Size         : 32
Epochs             : 10
Optimizer          : Adam
Learning Rate      : 0.0001
Loss Function      : Categorical Crossentropy
Activation         : Softmax

The pretrained MobileNetV2 layers are initially frozen while the newly added classification layers are trained.

Model Evaluation

The trained model was evaluated on 2,698 test images.

Test Performance
Accuracy       : 98%
Precision      : 98%
Recall         : 98%
F1-Score       : 98%
Class-wise Performance
Class	Precision	Recall	F1-Score
Fresh Apple	0.97	0.98	0.97
Fresh Banana	1.00	1.00	1.00
Fresh Orange	0.99	0.97	0.98
Rotten Apple	0.95	0.98	0.96
Rotten Banana	1.00	1.00	1.00
Rotten Orange	0.98	0.95	0.96
Overall	0.98	0.98	0.98
Results

The training process generates the following results:

results/
│
├── accuracy.png
├── loss.png
├── confusion_matrix.png
└── classification_report.txt
Accuracy Graph

The accuracy graph shows the training and validation accuracy across the training epochs.

Loss Graph

The loss graph shows the training and validation loss across the training epochs.

Confusion Matrix

The confusion matrix shows how accurately the model distinguishes between the six fruit/freshness categories.

Classification Report

The classification report provides:

Precision
Recall
F1-score
Support

for every class.

Example Prediction

The trained model can classify a new image from the user's computer.

Example:

Enter image path:
> C:\Users\ravip\Pictures\banana.jpg

The system produces a result such as:

==================================================
             AGROFRESH RESULT
==================================================
Food         : Apple
Condition    : ROTTEN
Confidence   : 91.52%
AI Class     : rottenapples
==================================================

The prediction above demonstrates that the system can analyze a completely new image outside the training and testing folders.

The result also highlights an important area for further development: real-world generalization. A model can achieve high accuracy on a prepared dataset but may behave differently on photographs captured under different lighting, backgrounds, camera angles, and conditions.

Current Limitations

The current version is a prototype and has some limitations.

The model recognizes only three types of fruits.
The classification is limited to fresh and rotten categories.
The training dataset may not represent all real-world lighting and environmental conditions.
Images captured using different cameras may produce different results.
The current model does not estimate the remaining shelf life of a fruit.
The current model does not provide a quantitative freshness score.
Further testing with independently captured real-world images is required.
Future Scope

AgroFresh can be extended into a more comprehensive food quality assessment system.

Planned Improvements
Real-time camera-based detection
Graphical user interface
Support for additional fruits and vegetables
Improved data augmentation
Fine-tuning of MobileNetV2
Comparison with MobileNetV3, EfficientNet and other models
Freshness score from 0–100
Estimated remaining shelf life
Detection under different lighting conditions
Mobile application
Raspberry Pi/edge AI deployment
IoT-based temperature and humidity monitoring
Cloud-based monitoring
Food waste reduction analytics
Proposed Future Architecture
                     AGROFRESH
                         |
                         v
                  Camera / Image
                         |
                         v
                 Image Processing
                         |
                         v
                  AI Classification
                         |
              +----------+----------+
              |                     |
              v                     v
          Food Type             Freshness
              |                     |
              +----------+----------+
                         |
                         v
                  Freshness Score
                         |
                         v
                Shelf-Life Estimate
                         |
                         v
                  User Dashboard
Project Structure
AgroFresh/
│
├── train.py
├── predict.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│
└── results/

The dataset is intentionally kept outside the GitHub repository because of its large number of image files.

Technologies Used
Python
TensorFlow
Keras
MobileNetV2
NumPy
Matplotlib
Scikit-learn
Pillow
Deep Learning
Computer Vision
Transfer Learning
How to Run
1. Install dependencies
pip install -r requirements.txt
2. Configure the dataset

The dataset should contain:

train/
├── freshapples/
├── freshbanana/
├── freshoranges/
├── rottenapples/
├── rottenbanana/
└── rottenoranges/

test/
├── freshapples/
├── freshbanana/
├── freshoranges/
├── rottenapples/
├── rottenbanana/
└── rottenoranges/

The dataset path is configured in:

config.py
3. Train the model
python train.py

The trained model will be saved in:

models/
4. Predict a new image
python predict.py

Enter the complete path of an image when prompted:

Enter image path:
> C:\Users\ravip\Pictures\banana.jpg
Project Status

Current Status: Prototype – Version 1

The initial MobileNetV2-based classification model has successfully been trained and evaluated.

The next development stage focuses on improving real-world performance using independently captured fruit images and improving the separation between fruit identification and freshness detection.

Objective

The long-term objective of AgroFresh is to develop an affordable and accessible AI-based food quality assessment system that can help households, retailers, restaurants, and food businesses identify potentially spoiled food and reduce unnecessary food wastage.

Disclaimer

AgroFresh is an academic and research prototype. Its predictions are based on visual characteristics learned from the training dataset and should not be considered a substitute for professional food safety inspection.


### One recommendation before you put this on GitHub

I deliberately wrote the README to **honestly report the current model** rather than claiming that AgroFresh is already production-ready.

In particular, the section:

> **Current Limitations → real-world generalization**

is important because your first real-world banana test was classified as **Rotten Apple at 91.52% confidence**. That is valuable information, and documenting it makes the project look **more scientifically credible**, not weaker.

I would **not put the "98% accuracy" in the project title or advertise it as real-world accuracy**. It is specifically the test-set accuracy of your current dataset.

Next, we should fix the fruit-identification problem and then update this README with the improved results.
                         |
                         v
                  Final Prediction
