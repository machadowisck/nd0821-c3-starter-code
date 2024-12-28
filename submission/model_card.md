# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A simple Random Forest prediction Model to evaluate income data. 
Parameters were not tuned, the default sklearn one swere used instead. 

## Intended Use
The model may be applyied to workers income information.

## Training Data
The data is the publicly available Census Bureau dataset is used for training and evaluating the model.
The sample used to train and test the model was obtained from the University of California at Irvine repository (https://archive.ics.uci.edu/dataset/20/census+income)

## Evaluation Data
The original dataset was first preprocessed and then split into training(80%) and evaluation(20%) data.

## Metrics
3 metrics were used for evaluating the model's performance: precision, recall, and f1. 

The model performance is as follows:

    - precision: 0.738
    - recall: 0.623
    - f1: 0.675


## Ethical Considerations
The Dataset is available under a This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

To learn more about the beforementioned license, please visit https://creativecommons.org/licenses/by/4.0/legalcode.

To the best of my knowledge, there are no known information in the dataset that could lead to individual identificaation. Still, careful usage of census data is always advised.

No further ethical concerns were identified within the present model scope.

## Caveats and Recommendations
The training dataset (Adult census) contains missing values and undefined values on the categorical features. Plan and adjust your prediction input  data accordingly. 
