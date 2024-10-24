import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

df= pd.read_csv('C:/Users/acer/Bank Loan/Model Build/train.csv')


x= df.drop('Loan_Status', axis='columns')
y = df['Loan_Status']

model = LogisticRegression(max_iter=1000)
model.fit(x,y)


pickle.dump(model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
