import keras
import numpy as np 
import pandas as pd  
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score


data=pd.read_csv(r"/Users/USER/Desktop/金融/HW6/2330.csv",sep=None,engine="python")
df2=pd.DataFrame(data[['%K','%D', 'DIF', 'MACD']])
df3=pd.DataFrame(data[['y']])

#,'MACD','DIF'

df3[df3 <= 0] = 0		  #將小於等於0的資料改為0	
df3[df3 > 0 ]=1		  #將大於0的資料改為1
X=df2[0:1200]		 #取0~1200筆資料
y=df3[0:1200]
Xtrain=df2[51:1200]	  #取51~1200筆資料	
ytrain=df3[51:1200]
Xtest=df2[0:50]		 #取0~50筆資料
ytest=df3[0:50]

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)  #建立模組
clf.fit(Xtrain,ytrain)						#套入資料

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)    		  #建立模組
kn.fit(Xtrain,ytrain)						#套入資料

from sklearn.svm import SVC
sv = SVC(kernel='rbf',gamma='auto') 			#建立模組
sv.fit(Xtrain,ytrain)						#套入資料

from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(random_state=0)	#建立模組
dc.fit(Xtrain,ytrain)		#套入資料

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()		#建立模組
gnb.fit(Xtrain,ytrain)		#套入資料

print("隨機森林",classification_report(ytest,clf.predict(Xtest)))
print("KNN",classification_report(ytest,kn.predict(Xtest)))
print("SVM",classification_report(ytest,sv.predict(Xtest)))
print("決策樹",classification_report(ytest,dc.predict(Xtest)))
print("Bayes",classification_report(ytest,gnb.predict(Xtest)))

s1 = cross_val_score(clf, X, y, cv=10)   #帶入交叉驗證套件
                                          #CV=資料分成份數
print(s1)
print("平均:",s1.mean())

s2 = cross_val_score(kn, X, y, cv=10)
print(s2)
print("平均:",s2.mean())

s3 = cross_val_score(sv, X, y, cv=10)
print(s3)
print("平均:",s3.mean())

s4 = cross_val_score(dc, X, y, cv=10)
print(s4)
print("平均:",s4.mean())

s5 = cross_val_score(gnb, X, y, cv=10)
print(s5)
print("平均:",s5.mean())

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation ,Flatten
from keras.utils import np_utils
from keras.utils import to_categorical 

data=pd.read_csv(r"/Users/USER/Desktop/金融/HW6/2330.csv",sep=None,engine="python")
df2=pd.DataFrame(data[['%K','%D', 'DIF', 'MACD']])
df3=pd.DataFrame(data[['y']])


df3[df3 <= 0] = 0		  #將小於等於0的資料改為0	
df3[df3 > 0 ]=1		  #將大於0的資料改為1
X=df2[0:1200]		  #取0~1200筆資料
y=df3[0:1200]
Xtrain=df2[51:1200]	  #取51~1200筆資料	
ytrain=df3[51:1200]
Xtest=df2[0:50]
ytest=df3[0:50]



model = Sequential()
model.add(Dense(10, input_dim=4,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation='sigmoid'))
model.add(Dropout(0.5))

from keras.optimizers import RMSprop
RMS=keras.optimizers.RMSprop(lr=0.0001,rho=0.9,epsilon=None,decay=0.0)   #建立優化器: 透過改變神經元權重使loss降低
model.compile(loss='sparse_categorical_crossentropy',optimizer=RMS,metrics=['accuracy'])  #建立模型
train_history = model.fit(x=Xtrain,y=ytrain,validation_split=0.2,epochs=80, batch_size=32, verbose=2)   #帶入資料

scores = model.evaluate(Xtest, ytest) 
print(scores) 
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))	#顯示正確率
prediction = model.predict_classes(Xtest)

from sklearn.metrics import confusion_matrix
conf_mx=confusion_matrix(ytest,prediction)      #帶入正確率套件
print(conf_mx)

tp=conf_mx[0,0]
fn=conf_mx[0,1]
fp=conf_mx[1,0]
tn=conf_mx[1,1]

print('Precision',tp/(tp+fp))
print('recall  ',tp/(tp+fn))
