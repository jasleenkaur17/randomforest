
"""
CS634-101 Data Mining --> Random Forest on Indian Liver Patient Dataset
Professor Byron 
Student: Karan Singh

References:
1) https://www.datascience.com/resources/notebooks/random-forest-intro
2) https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import random
from sklearn.model_selection import KFold
plt.show(block=False)

"""
Prepare Dataset:

1) Changed the Gender to Binary 
2) Changed the Outputs from 1 and 2 ---> to the appropriate format
3) Check the dataset for null values and then fill in the missing value with a mean , by using Imputer method
"""
dataset = pd.read_csv("/Users/karansingh/Documents/CS505 Random Forests/indian_liver_patient.csv")

mod_dataset = dataset
mod_dataset['Gender'] = mod_dataset['Gender'].map({"Male": int(1), "Female" : int(0)})
mod_dataset['Dataset'] = mod_dataset['Dataset'].map({int(1): int(1), int(2): int(0)})

mod_dataset.isnull().any()
mod_dataset.head()
mod_dataset.replace(r'\s+', np.nan, regex=True)

X = mod_dataset.iloc[:, 0:10].values
imp = Imputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)
Y = mod_dataset.iloc[:,10].values


""" 
Sckit Learn is being implemented to the split the Training Testing Sets:
"""

# The following code divides data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33,random_state =42)



"""
HyperParameters Optimization is beining used to find the best parameters for the Random Forest Classifier.
The Grid SeatchCV method is import from scikit learn
"""
# Set the random state for reproducbility
randomforestFit = RandomForestClassifier(random_state = 42)

# Hyperparameters Optimization
hyperparamSelec = {'max_depth': [2, 3, 4], 'bootstrap': [True, False], 'max_features': ['auto', 'sqrt', 'log2', None], 'criterion': ['gini', 'entropy']}

gridCV = GridSearchCV(randomforestFit, cv = 10, param_grid=hyperparamSelec, n_jobs = 3)

gridCV.fit(X_train, y_train)

#Set best parameters given by grid search 
randomforestFit.set_params(criterion = 'gini', max_features = 'auto', max_depth = 4)

"""
Calculate and Plot the Out-Of-Bag Error Rate
"""
randomforestFit.set_params(warm_start = True, oob_score = True)

error_calc = {}
minimumEstimate = 10
maxEstimate = 1000

for i in range(minimumEstimate, maxEstimate + 1):
    
    randomforestFit.set_params(n_estimators = i)
    randomforestFit.fit(X_train, y_train)
    
    oobError = 1 - randomforestFit.oob_score_
    error_calc[i] = oobError

#Convert Dictionary to a pandas series for easy plotting
oob = pd.Series(error_calc)

fig, ax = plt.subplots(figsize=(10,10))

oob.plot(kind='line',
                color = 'blue')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across selected band')
plt.show()

randomforestFit.set_params(n_estimators=300,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)


"""
Feature Importance
"""
randomforestFit.fit(X_train,y_train)
randomforestFit.feature_importances_
feature_importance = pd.Series(randomforestFit.feature_importances_, index = mod_dataset.columns[0:10].sort_values(ascending=False))
#Creating a bar plot
sns.barplot(x=feature_importance, y=feature_importance.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

"""
K-Fold Cross Validation
"""
def cross_val_metrics(fit, training_set, class_set, estimator, print_results = True):
    """
    Purpose
    ----------
    Function helps automate cross validation processes while including
    option to print metrics or store in variable

    Parameters
    ----------
    fit: Fitted model
    training_set:  Data_frame containing 80% of original dataframe
    class_set:     data_frame containing the respective target vaues
                      for the training_set
    print_results: Boolean, if true prints the metrics, else saves metrics as
                      variables

    Returns
    ----------
    scores.mean(): Float representing cross validation score
    scores.std() / 2: Float representing the standard error (derived
                from cross validation score's standard deviation)
    """
    my_estimators = {
    'rf': 'estimators_',
    'nn': 'out_activation_',
    'knn': '_fit_method'
    }
    try:
        # Captures whether first parameter is a model
        if not hasattr(fit, 'fit'):
            return print("'{0}' is not an instantiated model from scikit-learn".format(fit))

        # Captures whether the model has been trained
        if not vars(fit)[my_estimators[estimator]]:
            return print("Model does not appear to be trained.")

    except KeyError as e:
        raise("'{0}' does not correspond with the appropriate key inside the estimators dictionary.               Please refer to function to check `my_estimators` dictionary.".format(estimator))

    n = KFold(n_splits=10)
    scores = cross_val_score(fit,
                         training_set,
                         class_set,
                         cv = n)
    if print_results:
        for i in range(0, len(scores)):
            print("Cross validation run {0}: {1: 0.3f}".format(i, scores[i]))
        print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"              .format(scores.mean(), scores.std() / 2))
    else:
        return scores.mean(), scores.std() / 2


cross_val_metrics(randomforestFit, 
                  X_train, 
                  y_train, 
                  'rf',
                  print_results = True)


# ROC Curve Calculations
predictProbab = randomforestFit.predict_proba(X_test)[:, 1]
f, t, _ = roc_curve(y_test, predictProbab, pos_label = 1)
aucr = auc(f, t)


# Function is used to plot the ROC Curve
def plot_roc_curve(fpr, tpr, auc, estimator, xlim=None, ylim=None):
    """
    Purpose
    ----------
    Function creates ROC Curve for respective model given selected parameters.
    Optional x and y limits to zoom into graph

    Parameters
    ----------
    * fpr: 	Array returned from sklearn.metrics.roc_curve for increasing
    false positive rates
    * tpr: 	Array returned from sklearn.metrics.roc_curve for increasing
    true positive rates
    * auc:	Float returned from sklearn.metrics.auc (Area under Curve)
    * estimator: 	String represenation of appropriate model, can only contain the
    following: ['knn', 'rf', 'nn']
    * xlim:		Set upper and lower x-limits
    * ylim:		Set upper and lower y-limits
    """
    my_estimators = {'knn': ['Kth Nearest Neighbor', 'deeppink'],
              'rf': ['Random Forest', 'red'],
              'nn': ['Neural Network', 'purple']}

    try:
        plot_title = my_estimators[estimator][0]
        color_value = my_estimators[estimator][1]
    except KeyError as e:
        raise("'{0}' does not correspond with the appropriate key inside the estimators dictionary.               Please refer to function to check `my_estimators` dictionary.".format(estimator))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#fafafa')

    plt.plot(fpr, tpr,
             color=color_value,
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'              .format(plot_title, auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
    plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
    plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()


plot_roc_curve(f, t, aucr, 'rf',
               xlim=(-0.01, 1.05), 
               ylim=(0.001, 1.05))


xval = ['Benign', 'Malignant']

def print_class_report(predictions, alg_name):
    """
    Purpose
    ----------
    Function helps automate the report generated by the
    sklearn package. Useful for multiple model comparison

    Parameters:
    ----------
    predictions: The predictions made by the algorithm used
    alg_name: String containing the name of the algorithm used
    
    Returns:
    ----------
    Returns classification report generated from sklearn. 
    """
    print('Classification Report for {0}:'.format(alg_name))
    print(classification_report(predictions, 
            y_test, 
            target_names = xval))



# Best Parameters using the Grid Seatch
print(" ")
print('Best Parameters using grid search: \n', 
      gridCV.best_params_)

#OOB Error Rate
print(" ")
print('OOB Error rate for 300 trees is: {0:.5f}'.format(oob[300]))
print(" ")

#Confusion Matrix
predictrandFor = randomforestFit.predict(X_test)
print("The Confusion Matrix")
print(confusion_matrix(y_test,predictrandFor))
print(" ")

# Mean Acccuracy on the test set
accuracy_rf = randomforestFit.score(X_test, y_test)
print("Mean accuracy on the test set:\n {0:.3f}".format(accuracy_rf))
print(" ")

# Calculate the Test Error Rate and compare against OOB
test_error_rate_rf = 1 - accuracy_rf
print("Test Error Rate for our model:\n {0: .4f}".format(test_error_rate_rf))
print(" ")

#Classification Report
class_report = print_class_report(predictrandFor, 'Random Forest')
print(accuracy_score(y_test, predictrandFor))

quit()
