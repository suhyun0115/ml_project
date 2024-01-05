#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nbconvert')


# In[2]:


jupyter nbconvert --to script final_all_ML_GSCV.ipynb


# In[ ]:


import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 서포트 벡터 머신에 대한 그리드 서치 파라미터
svc_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['auto', 0.1, 1, 10],
    'max_iter': [1000, 2000, 3000]
}

# 결과를 저장할 리스트 초기화
all_reports_svc = []

# 각 데이터프레임에 대해 서포트 벡터 머신 모델 학습과 평가
for i in range(1, 7):
    train_csv_path = f'train_data_{i}.csv'
    test_csv_path = f'test_data_{i}.csv'

    print(f"\n---- Training and Evaluating SVC for {train_csv_path} ----")

    # CSV 파일을 데이터프레임으로 읽기
    df_train = pd.read_csv(train_csv_path)

    # X, y 설정
    y_train = df_train['당뇨유무']
    X_train = df_train.drop('당뇨유무', axis=1)

    # 대응하는 test 데이터셋
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop('당뇨유무', axis=1)
    y_test = df_test['당뇨유무']

    # 서포트 벡터 머신에 대한 그리드 서치
    svc_model = SVC()
    svc_gscv = GridSearchCV(svc_model, svc_params, cv=5, scoring='f1_macro')

    # 모델 학습
    svc_gscv.fit(X_train, y_train)

    # 결과 출력
    print(f"SVC for {train_csv_path} - Best Parameters:", svc_gscv.best_params_)
    print(f"SVC for {train_csv_path} - Best F1 Score:", svc_gscv.best_score_)

    # 모델에 대한 분류 리포트 출력
    svc_report = classification_report(y_test, svc_gscv.predict(X_test))
    print(f"SVC Classification Report for {test_csv_path} (on Test Set):\n", svc_report)

    # 리포트를 리스트에 추가
    all_reports_svc.append(svc_report)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# K-최근접 이웃에 대한 그리드 서치 파라미터
knn_params = {
    'n_neighbors': [3, 5, 7, 10, 15, 20, 25],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# 결과를 저장할 리스트 초기화
all_reports_knn = []

# 각 데이터프레임에 대해 K-최근접 이웃 모델 학습과 평가
for i in range(1, 7):
    train_csv_path = f'train_data_{i}.csv'
    test_csv_path = f'test_data_{i}.csv'

    print(f"\n---- Training and Evaluating KNN for {train_csv_path} ----")

    # CSV 파일을 데이터프레임으로 읽기
    df_train = pd.read_csv(train_csv_path)

    # X, y 설정
    y_train = df_train['당뇨유무']
    X_train = df_train.drop('당뇨유무', axis=1)

    # 대응하는 test 데이터셋
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop('당뇨유무', axis=1)
    y_test = df_test['당뇨유무']

    # K-최근접 이웃에 대한 그리드 서치
    knn_model = KNeighborsClassifier()
    knn_gscv = GridSearchCV(knn_model, knn_params, cv=5, scoring='f1_macro')

    # 모델 학습
    knn_gscv.fit(X_train, y_train)

    # 결과 출력
    print(f"KNN for {train_csv_path} - Best Parameters:", knn_gscv.best_params_)
    print(f"KNN for {train_csv_path} - Best F1 Score:", knn_gscv.best_score_)

    # 모델에 대한 분류 리포트 출력
    knn_report = classification_report(y_test, knn_gscv.predict(X_test))
    print(f"KNN Classification Report for {test_csv_path} (on Test Set):\n", knn_report)

    # 리포트를 리스트에 추가
    all_reports_knn.append(knn_report)


# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# XGBoost에 대한 그리드 서치 파라미터
xgb_params = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 3],
    'gamma': [0, 1, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'n_jobs': [-1]
}


# 결과를 저장할 리스트 초기화
all_reports_xgb = []

# 각 데이터프레임에 대해 XGBoost 모델 학습과 평가
for i in range(1, 7):
    train_csv_path = f'train_data_{i}.csv'
    test_csv_path = f'test_data_{i}.csv'

    print(f"\n---- Training and Evaluating XGBoost for {train_csv_path} ----")

    # CSV 파일을 데이터프레임으로 읽기
    df_train = pd.read_csv(train_csv_path)

    # X, y 설정
    y_train = df_train['당뇨유무']
    X_train = df_train.drop('당뇨유무', axis=1)

    # 대응하는 test 데이터셋
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop('당뇨유무', axis=1)
    y_test = df_test['당뇨유무']

    # XGBoost에 대한 그리드 서치
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    xgb_gscv = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='f1_macro')

    # 모델 학습
    xgb_gscv.fit(X_train, y_train)

    # 결과 출력
    print(f"XGBoost for {train_csv_path} - Best Parameters:", xgb_gscv.best_params_)
    print(f"XGBoost for {train_csv_path} - Best F1 Score:", xgb_gscv.best_score_)

    # 모델에 대한 분류 리포트 출력
    xgb_report = classification_report(y_test, xgb_gscv.predict(X_test))
    print(f"XGBoost Classification Report for {test_csv_path} (on Test Set):\n", xgb_report)

    # 리포트를 리스트에 추가
    all_reports_xgb.append(xgb_report)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Adaboost에 대한 그리드 서치 파라미터
adaboost_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'base_estimator': [
        DecisionTreeClassifier(criterion='gini', max_features=7, max_depth=1),
        DecisionTreeClassifier(criterion='gini', max_features=7, max_depth=2),
        DecisionTreeClassifier(criterion='entropy', max_features=7, max_depth=1),
        DecisionTreeClassifier(criterion='entropy', max_features=7, max_depth=2),
        DecisionTreeClassifier(criterion='gini', max_features=8, max_depth=1),
        DecisionTreeClassifier(criterion='gini', max_features=8, max_depth=2),
        DecisionTreeClassifier(criterion='entropy', max_features=8, max_depth=1),
        DecisionTreeClassifier(criterion='entropy', max_features=8, max_depth=2)
    ],
}


# 결과를 저장할 리스트 초기화
all_reports_adaboost = []

# 각 데이터프레임에 대해 Adaboost 모델 학습과 평가
for i in range(1, 7):
    train_csv_path = f'train_data_{i}.csv'
    test_csv_path = f'test_data_{i}.csv'

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

    # Adaboost에 대한 그리드 서치
    adaboost_model = AdaBoostClassifier()
    adaboost_gscv = GridSearchCV(adaboost_model, adaboost_params, cv=5, scoring='f1_macro')

    # 모델 학습
    adaboost_gscv.fit(X_train, y_train)

    # 결과 출력
    print(f"Adaboost for {train_csv_path} - Best Parameters:", adaboost_gscv.best_params_)
    print(f"Adaboost for {train_csv_path} - Best F1 Score:", adaboost_gscv.best_score_)

    # 모델에 대한 분류 리포트 출력
    adaboost_report = classification_report(y_test, adaboost_gscv.predict(X_test))
    print(f"Adaboost Classification Report for {test_csv_path} (on Test Set):\n", adaboost_report)

    # 리포트를 리스트에 추가
    all_reports_adaboost.append(adaboost_report)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Naive Bayes에 대한 그리드 서치 파라미터
nb_params = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}

# 결과를 저장할 리스트 초기화
all_reports_nb = []

# 각 데이터프레임에 대해 Naive Bayes 모델 학습과 평가
for i in range(1, 7):
    train_csv_path = f'train_data_{i}.csv'
    test_csv_path = f'test_data_{i}.csv'

    print(f"\n---- Training and Evaluating Naive Bayes for {train_csv_path} ----")

    # CSV 파일을 데이터프레임으로 읽기
    df_train = pd.read_csv(train_csv_path)

    # X, y 설정
    y_train = df_train['당뇨유무']
    X_train = df_train.drop('당뇨유무', axis=1)

    # 대응하는 test 데이터셋
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop('당뇨유무', axis=1)
    y_test = df_test['당뇨유무']

    # Naive Bayes에 대한 그리드 서치
    nb_model = GaussianNB()
    nb_gscv = GridSearchCV(nb_model, nb_params, cv=5, scoring='f1_macro')

    # 모델 학습
    nb_gscv.fit(X_train, y_train)

    # 결과 출력
    print(f"Naive Bayes for {train_csv_path} - Best Parameters:", nb_gscv.best_params_)
    print(f"Naive Bayes for {train_csv_path} - Best F1 Score:", nb_gscv.best_score_)

    # 모델에 대한 분류 리포트 출력
    nb_report = classification_report(y_test, nb_gscv.predict(X_test))
    print(f"Naive Bayes Classification Report for {test_csv_path} (on Test Set):\n", nb_report)

    # 리포트를 리스트에 추가
    all_reports_nb.append(nb_report)

# 모든 모델의 리포트 출력
print("\nAll Classification Reports for Naive Bayes:")
for i, report in enumerate(all_reports_nb, start=1):
    print(f"Classification Report for df_{i}:\n", report)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Random Forest에 대한 그리드 서치 파라미터
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'n_jobs': [-1]
}

# 결과를 저장할 리스트 초기화
all_reports_rf = []

# 각 데이터프레임에 대해 Random Forest 모델 학습과 평가
for i in range(1, 7):
    train_csv_path = f'train_data_{i}.csv'
    test_csv_path = f'test_data_{i}.csv'

    print(f"\n---- Training and Evaluating Random Forest for {train_csv_path} ----")

    # CSV 파일을 데이터프레임으로 읽기
    df_train = pd.read_csv(train_csv_path)

    # X, y 설정
    y_train = df_train['당뇨유무']
    X_train = df_train.drop('당뇨유무', axis=1)

    # 대응하는 test 데이터셋
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop('당뇨유무', axis=1)
    y_test = df_test['당뇨유무']

    # Random Forest에 대한 그리드 서치
    rf_model = RandomForestClassifier()
    rf_gscv = GridSearchCV(rf_model, rf_params, cv=5, scoring='f1_macro')

    # 모델 학습
    rf_gscv.fit(X_train, y_train)

    # 결과 출력
    print(f"Random Forest for {train_csv_path} - Best Parameters:", rf_gscv.best_params_)
    print(f"Random Forest for {train_csv_path} - Best F1 Score:", rf_gscv.best_score_)

    
    # 모델에 대한 분류 리포트 출력
    rf_report = classification_report(y_test, rf_gscv.predict(X_test))
    print(f"Random Forest Classification Report for {test_csv_path} (on Test Set):\n", rf_report)

    # 리포트를 리스트에 추가
    all_reports_rf.append(rf_report)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Decision Tree에 대한 그리드 서치 파라미터
dt_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# 결과를 저장할 리스트 초기화
all_reports_dt = []

# 각 데이터프레임에 대해 Decision Tree 모델 학습과 평가
for i in range(1, 7):
    train_csv_path = f'train_data_{i}.csv'
    test_csv_path = f'test_data_{i}.csv'

    print(f"\n---- Training and Evaluating Decision Tree for {train_csv_path} ----")

    # CSV 파일을 데이터프레임으로 읽기
    df_train = pd.read_csv(train_csv_path)

    # X, y 설정
    y_train = df_train['당뇨유무']
    X_train = df_train.drop('당뇨유무', axis=1)

    # 대응하는 test 데이터셋
    df_test = pd.read_csv(test_csv_path)
    X_test = df_test.drop('당뇨유무', axis=1)
    y_test = df_test['당뇨유무']

    # Decision Tree에 대한 그리드 서치
    dt_model = DecisionTreeClassifier()
    dt_gscv = GridSearchCV(dt_model, dt_params, cv=5, scoring='f1_macro')

    # 모델 학습
    dt_gscv.fit(X_train, y_train)

    # 결과 출력
    print(f"Decision Tree for {train_csv_path} - Best Parameters:", dt_gscv.best_params_)
    print(f"Decision Tree for {train_csv_path} - Best F1 Score:", dt_gscv.best_score_)

    # 모델에 대한 분류 리포트 출력
    dt_report = classification_report(y_test, dt_gscv.predict(X_test))
    print(f"Decision Tree Classification Report for {test_csv_path} (on Test Set):\n", dt_report)

    # 리포트를 리스트에 추가
    all_reports_dt.append(dt_report)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




