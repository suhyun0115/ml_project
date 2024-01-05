#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[6]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
import os
import pandas as pd

adaboost_model = joblib.load('./ada_model.pkl')

# 초기값 설정
i = 1
train_csv_path = f'train_data_{i}.csv'
test_csv_path = f'test_data_{i}.csv'

# 각 데이터프레임에 대해 Adaboost 모델 학습과 평가
while os.path.exists(train_csv_path) and os.path.exists(test_csv_path):
    print(f"\n---- Training and Evaluating Adaboost for {train_csv_path} ----")

    # CSV 파일을 데이터프레임으로 읽기
    df_train = pd.read_csv(train_csv_path)

    # X, y 설정
    y_train = df_train['당뇨유무']
    X_train = df_train.drop('당뇨유무', axis=1)

    # 대응하는 test 데이터셋
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop('당뇨유무', axis=1)
    y_test = df_test['당뇨유무']

    # 모델에 대한 분류 리포트 출력
    adaboost_report = classification_report(y_test, adaboost_model.predict(X_test))
    print(f"Adaboost Classification Report for {test_csv_path} (on Test Set):\n", adaboost_report)

    # Update values for the next iteration
    i += 1
    train_csv_path = f'train_data_{i}.csv'
    test_csv_path = f'test_data_{i}.csv'


# In[ ]:




