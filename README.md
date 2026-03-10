# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Python libraries and load the food dataset for diabetic classification. 

2.Load the dataset containing food items and their nutritional information. 

3.Train the SVM classifier using the training dataset and tune the hyperparameters to improve performance.

4.Test the model using the test dataset and display the classification results and accuracy. 

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
import seaborn as sns

data=pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)
features=['Calories','Total Fat', 'Saturated Fat', 'Sugars','Dietary Fiber','Protein' ]
target='class'

X=data[features]
y=data[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

svm=SVC()
param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']}
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
best_model=grid_search.best_estimator_

print("Name: KRITHIKAA P")
print("Register Number:212225040193")
print("Best Parameters:",grid_search.best_params_)

y_pred=best_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name: KRITHIKAA P")
print("Register Number:212225040193")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="1039" height="705" alt="Screenshot 2026-03-10 080225" src="https://github.com/user-attachments/assets/8122fb3d-19d5-417f-b187-3f6e1836869c" />

<img width="1055" height="84" alt="Screenshot 2026-03-10 080238" src="https://github.com/user-attachments/assets/02e7f62f-d33f-44c5-bc20-8a714644712e" />

<img width="995" height="292" alt="Screenshot 2026-03-10 080252" src="https://github.com/user-attachments/assets/c6d57ac1-9cf9-435d-8003-9338b172af77" />

<img width="897" height="560" alt="Screenshot 2026-03-10 080306" src="https://github.com/user-attachments/assets/5a9c19c1-ca1f-43e6-98fe-bce10b22b148" />


## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
