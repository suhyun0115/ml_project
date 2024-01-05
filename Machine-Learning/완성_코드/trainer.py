import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_save_models():
    # Initialize variables to keep track of the best model and its accuracy
    best_accuracy = 0
    best_model = None
    best_length = 0

    # AdaBoost 모델 생성
    adaboost_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=1, max_features=8),
        learning_rate=0.1,
        n_estimators=100
    )

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

        # 모델 학습
        adaboost_model.fit(X_train, y_train)

        # 모델에 대한 평가
        y_pred = adaboost_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{i}번째 {accuracy}')

        # 모델이 이전에 저장한 모델보다 더 나은 경우 갱신
        if (accuracy > best_accuracy) and (len(df_train) >= best_length):
            best_accuracy = accuracy
            best_model = adaboost_model
            best_length = len(df_train)
            print(f"Best Model Updated! Accuracy: {best_accuracy}")

        i += 1
        train_csv_path = f'train_data_{i}.csv'
        test_csv_path = f'test_data_{i}.csv'

    if best_model is not None:
        # 저장된 최적의 모델을 파일로 저장
        joblib.dump(best_model, 'ada_best_model.pkl')
        print("Best Model Saved Successfully!")
        print(best_accuracy)
    else:
        print("No models met the criteria for saving.")

if __name__ == "__main__":
    # Call the function if the module is executed as a script
    train_and_save_models()
