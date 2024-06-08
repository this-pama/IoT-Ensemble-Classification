
# Ensemble Learning and Deep Neural Networks for Classification

This project explores various ensemble learning techniques and deep neural networks (DNN) for classification tasks using the CICIoT2023 dataset. The methods include AdaBoost, Random Forest, Bagging, and Stacking with combinations of classifiers like KNN, SVM, Logistic Regression, and DNN.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Models Implemented](#models-implemented)
  - [AdaBoost](#adaboost)
  - [Random Forest](#random-forest)
  - [Random Subspace Method](#random-subspace-method)
  - [Bagging with RF and DNN](#bagging-with-rf-and-dnn)
  - [Stacking](#stacking)
  - [Deep Neural Network](#deep-neural-network)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Run](#how-to-run)

## Project Overview

The goal of this project is to evaluate the performance of various ensemble learning methods and a deep neural network on a classification task. The methods are implemented in Jupyter notebooks and tested on the CICIoT2023 dataset.

## Dataset

The dataset used for this project is the CICIoT2023, which can be found [here](https://www.unb.ca/cic/datasets/iotdataset-2023.html). This dataset is provided by the Canadian Institute for Cybersecurity (CIC) and contains labeled data for IoT device activities, which is used for training and evaluating machine learning models in this project.

## Data Loading and Preprocessing

Data is read from CSV files in both training and testing folders. The steps include:

1. **Reading Data**: Using a custom function to read CSV files.
2. **Encoding Categorical Variables**: Encoding labels using `LabelEncoder`.
3. **Splitting Data**: Splitting data into features and target variable.
4. **Scaling Data**: Standardizing features using `StandardScaler`.

## Models Implemented

### AdaBoost

Implemented using `AdaBoostClassifier` with `DecisionTreeClassifier` as the base estimator. The model handles class imbalance by computing class weights.

### Random Forest

Implemented using `RandomForestClassifier` with 100 estimators.

### Random Subspace Method

Implemented using `BaggingClassifier` with `DecisionTreeClassifier` as the base learner and random subspaces.

### Bagging with RF and DNN

Implemented bagging with both Random Forest and a Deep Neural Network (DNN). The DNN is created using `Sequential` from Keras.

### Stacking

Implemented stacking with various combinations of classifiers:
- `knn`, `rf`, `dnn`, `LogisticRegression`
- `knn`, `svm`, `LogisticRegression`

### Deep Neural Network

A standalone DNN with two hidden layers and a softmax output layer, trained using `Sequential` from Keras.

## Evaluation

Models are evaluated using accuracy, classification reports, and confusion matrices. Training history plots are also provided for the DNN models.

## Results

- **AdaBoost**: Provides accuracy and detailed classification metrics.
- **Random Forest**: Outputs accuracy and confusion matrix.
- **Random Subspace Method**: Evaluates accuracy and classification metrics.
- **Bagging with RF and DNN**: Compares performance of individual and combined models.
- **Stacking**: Evaluates stacking model accuracy and classification metrics.
- **Deep Neural Network**: Plots training history, evaluates test accuracy, and provides classification metrics.

## How to Run

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/ensemble-dnn-classification.git
    cd ensemble-dnn-classification
    ```

2. **Set Up Environment**:
    Install necessary libraries (consider using a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Jupyter Notebooks**:
    Launch Jupyter Notebook and run the notebooks in sequence to reproduce the results:
    ```bash
    jupyter notebook
    ```

4. **Google Drive Integration**:
    Ensure your Google Drive is mounted correctly if you're using Google Colab.

