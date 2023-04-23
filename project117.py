import matplotlib.pyplot as plp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data = pd.read_csv('BankNote_Authentication.csv')
score = data[['variance','skewness','curtosis','entropy']]
result = data['class']

score_test,score_train,result_test,result_train = train_test_split(score,result,test_size=0.25,random_state=0)

standard = StandardScaler()
score_train = standard.fit_transform(score_train)
score_test = standard.fit_transform(score_test)

logreg = LogisticRegression()
logreg.fit(score_train,result_train)

error_prediction = logreg.predict(score_test)
error_value = []

for i in error_prediction:
    if(i == 0):
        error_value.append('No')
    else:
        error_value.append('Yes')

actual_value = []

for i in result_test.ravel():
    if(i == 0):
        actual_value.append('No')
    else:
        actual_value.append('Yes')

matrix = confusion_matrix(actual_value,error_value)
a = plp.subplot()

sns.heatmap(matrix,annot = True,ax=a)
plp.xlabel('Prediction')
plp.ylabel('Actual')
plp.show()

accuracy = accuracy_score(error_value,actual_value)
print('accuracy -->',accuracy*100)