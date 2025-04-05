# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The Random Forest model is used for modeling the income, the implementation is imported from scikit-learn.

## Intended Use

The model can be used to determine whether a person makes over 50K a year, based on the input features, characteristics related to corresponding person.

## Training Data

The training data is derived from https://archive.ics.uci.edu/dataset/20/census+income.
There are 14 feature and 1 label columns in the dataset, with over 32k of samples.
Categorical features were encoded with OneHotEncoder and label was encoded with LabelBinarizer for both training and evaluation data. 

## Evaluation Data

The preprocessing is the same as for trining data, the test size is 0.2. 

## Metrics

The output metrics on evaluation data:
Precision: 0.7291
Recall: 0.6403
Fbeta: 0.6818

## Ethical Considerations

The model's outputs should not be used misused to disciminate people based on sex, race or age. 

## Caveats and Recommendations

The further use of model and other deliverables based on data analysis can be used to research particular social effects. The current model or alternative can be improved using more thorough feature engineering and model parameters search.
