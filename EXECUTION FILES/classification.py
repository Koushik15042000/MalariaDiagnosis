import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
import joblib

##Step1: Load Dataset

dataframe = pd.read_csv("F://MINI-PROJECT//csv//dataset.csv")
#print(dataframe.head())

#Step2: Split into training and test data
x = dataframe.drop(["Label"],axis=1)
y = dataframe["Label"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#syntax: train_test_split(x,y,Percentage_Diivsion,)
#x_train.fillna(x_train.mean())
#y_train.fillna(y_train.mean())

##Step4: Build a model

model = RandomForestClassifier(n_estimators=100,max_depth=5)
model.fit(x_train,y_train)
joblib.dump(model,"rf_malaria_100_5")


##Step5: Make predictions and get classification report

predictions = model.predict(x_test)
#Precision -> Percent of correct Predictions
#Recall -> Percent of Positive Cases we caught
#f1-score -> Percent of Postive predictions that were correct

print(metrics.classification_report(predictions,y_test))
print(model.score(x_test,y_test))
















