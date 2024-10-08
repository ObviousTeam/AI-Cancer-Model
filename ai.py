import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

# Read the CSV file from a local path
dataset = pd.read_csv("data.csv")


Labels = dataset['diagnosis']
Features = dataset[['radius_mean',	'texture_mean',	'perimeter_mean',	'area_mean'	, 'smoothness_mean' ,	'compactness_mean' ,	'concavity_mean' ,	'concave points_mean',	'symmetry_mean',	'fractal_dimension_mean']]
train1, test1, train2, test2 = train_test_split(Features, Labels, test_size=.2, shuffle=True)

X = dataset[['radius_mean',	'texture_mean',	'perimeter_mean',	'area_mean'	, 'smoothness_mean']]
Y = dataset[['compactness_mean' ,	'concavity_mean' ,	'concave points_mean',	'symmetry_mean',	'fractal_dimension_mean']]
sns.pairplot(dataset, x_vars = X, y_vars = Y, hue = 'diagnosis', palette = 'Blues', kind = 'scatter')

Num_Pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# String_Pipeline = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

Preprocessor = ColumnTransformer(
    transformers=[
        ('num', Num_Pipeline, ['radius_mean',	'texture_mean',	'perimeter_mean',	'area_mean'	, 'smoothness_mean' ,	'compactness_mean' ,	'concavity_mean' ,	'concave points_mean',	'symmetry_mean',	'fractal_dimension_mean'])
        # ('cat', String_Pipeline, [])
    ])


classifiers = {
    'DecisionTree': (DecisionTreeClassifier(), {
        'max_depth': [10, 20, 30],
        'criterion' : ['gini', 'entropy'],
        'min_samples_split' : [2,3,4,5,6,7,8],
        'random_state' : [0,10,20,30,40,50,60,70,80]
    }),
    'RandomForest': (RandomForestClassifier(), {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [10, 20, 30],
        'criterion' : ['gini', 'entropy'],
        'min_samples_split' : [2,3,4,5,6,7,8]
    }),
    'LogisticRegression': (LogisticRegression(), {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 300],
        'l1_ratio': [0.5, 0.7]
    }),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [1,2,3,4,5,6,7,8,9,10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['brute', 'auto', 'ball_tree', 'kd_tree']
    })
}
test_score = 0
Classifier_Profile = None

for name, (clf, param_grid) in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(train1, train2)

    if grid_search.best_score_ > test_score:
        test_score = grid_search.best_score_
        Classifier_Profile = {
            'classifier': name,
            'score': grid_search.best_score_,
            'params': grid_search.best_params_,
            'estimator': grid_search.best_estimator_
        }



joblib.dump(Classifier_Profile['estimator'], './ai.pkl')
print(f"Best classifier saved: {Classifier_Profile['classifier']} with score: {Classifier_Profile['score']}")