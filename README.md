# Credit-Fraud-Detection-with-Data-Augmentation
***

## Task
The task here is to build and train prediction models 
for credit fraud detection using highly unbalanced data. 

## Description
The problem was solved by using four different ML models:
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `GradientBoostingClassifier`
all of which are from sklearn. Due to the unbalanced nature 
of the data, four datasets were used:
- control (uniformly sampled)
- SMOTE augmented 
- ADASYN augmented 
- GAN augmented 
SMOTE and ADASYN were implemented from the imblearn package, 
while the GAN was a custom PyTorch implementation. Each model 
was trained on each dataset. Metrics recorded were 
- precision
- recall
- f1 score 
- accuracy 
- area under ROC curve 
All models were saved after training for reproduceability. 

## Installation
The following Python 3 packages are required:
- `numpy`
- `pandas` 
- `pytorch` 
- `sklearn` 
- `pickle` 
To install each package, simply run `pip install <package>` in 
a Jupyter cell. 

## Usage
The file `a_fool_fraud_tradi.ipynb` uses the control (uniformly 
sampled) dataset, while `a_fool_fraud_data_augmentation.ipynb` 
contains the generation, training, and evaluation with the 
augmented datasets. In either notebook, simply run all cells to 
view the generation of the synthetic data and/or the evaluation 
of the models. Keeping the models files in the same directory will 
lead to 

### The Core Team
Andrew Bonham

<span><i>Made at <a href='https://qwasar.io'>Qwasar SV -- Software Engineering School</a></i></span>
<span><img alt="Qwasar SV -- Software Engineering School's Logo" src='https://storage.googleapis.com/qwasar-public/qwasar-logo_50x50.png' width='20px' /></span>
