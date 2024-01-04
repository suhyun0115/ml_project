{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "0aed2a14-d369-4e19-af3f-f4d810f42d55",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "---- Training and Evaluating Adaboost for train_data_1.csv ----\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\envs\\WC\\lib\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1번째 0.7947194719471947\n",
      "Best Model Updated! Accuracy: 0.7947194719471947\n",
      "\n",
      "---- Training and Evaluating Adaboost for train_data_2.csv ----\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\envs\\WC\\lib\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2번째 0.8118811881188119\n",
      "Best Model Updated! Accuracy: 0.8118811881188119\n",
      "\n",
      "---- Training and Evaluating Adaboost for train_data_3.csv ----\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\envs\\WC\\lib\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3번째 0.7920792079207921\n",
      "\n",
      "---- Training and Evaluating Adaboost for train_data_4.csv ----\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\envs\\WC\\lib\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4번째 0.803960396039604\n",
      "\n",
      "---- Training and Evaluating Adaboost for train_data_5.csv ----\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\envs\\WC\\lib\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5번째 0.7920792079207921\n",
      "\n",
      "---- Training and Evaluating Adaboost for train_data_6.csv ----\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\anaconda3\\envs\\WC\\lib\\site-packages\\sklearn\\ensemble\\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6번째 0.8139737991266376\n",
      "Best Model Updated! Accuracy: 0.8139737991266376\n",
      "Best Model Saved Successfully!\n"
     ]
    }
   ],
   "source": [
    "# create_models.py\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import AdaBoostClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "import joblib\n",
    "import os\n",
    "\n",
    "def train_and_save_models():\n",
    "    # Initialize variables to keep track of the best model and its accuracy\n",
    "    best_accuracy = 0\n",
    "    best_model = None\n",
    "    best_length = 0\n",
    "\n",
    "    # AdaBoost 모델 생성\n",
    "    adaboost_model = AdaBoostClassifier(\n",
    "        base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=1, max_features=8),\n",
    "        learning_rate=0.1,\n",
    "        n_estimators=100\n",
    "    )\n",
    "\n",
    "    # 초기값 설정\n",
    "    i = 1\n",
    "    train_csv_path = f'train_data_{i}.csv'\n",
    "    test_csv_path = f'test_data_{i}.csv'\n",
    "\n",
    "    # 각 데이터프레임에 대해 Adaboost 모델 학습과 평가\n",
    "    while os.path.exists(train_csv_path) and os.path.exists(test_csv_path):\n",
    "        print(f\"\\n---- Training and Evaluating Adaboost for {train_csv_path} ----\")\n",
    "\n",
    "        # CSV 파일을 데이터프레임으로 읽기\n",
    "        df_train = pd.read_csv(train_csv_path)\n",
    "\n",
    "        # X, y 설정\n",
    "        y_train = df_train['당뇨유무']\n",
    "        X_train = df_train.drop('당뇨유무', axis=1)\n",
    "\n",
    "        # 대응하는 test 데이터셋\n",
    "        df_test = pd.read_csv(test_csv_path)\n",
    "        X_test = df_test.drop('당뇨유무', axis=1)\n",
    "        y_test = df_test['당뇨유무']\n",
    "\n",
    "        # 모델 학습\n",
    "        adaboost_model.fit(X_train, y_train)\n",
    "\n",
    "        # 모델에 대한 평가\n",
    "        y_pred = adaboost_model.predict(X_test)\n",
    "        accuracy = accuracy_score(y_test, y_pred)\n",
    "        print(f'{i}번째 {accuracy}')\n",
    "\n",
    "        # 모델이 이전에 저장한 모델보다 더 나은 경우 갱신\n",
    "        if accuracy > best_accuracy or (accuracy == best_accuracy and len(df_train) >= best_length):\n",
    "            best_accuracy = accuracy\n",
    "            best_model = adaboost_model\n",
    "            best_length = len(df_train)\n",
    "            print(f\"Best Model Updated! Accuracy: {best_accuracy}\")\n",
    "\n",
    "        i += 1\n",
    "        train_csv_path = f'train_data_{i}.csv'\n",
    "        test_csv_path = f'test_data_{i}.csv'\n",
    "\n",
    "    if best_model is not None:\n",
    "        # 저장된 최적의 모델을 파일로 저장\n",
    "        joblib.dump(best_model, 'ada_best_model.pkl')\n",
    "        print(\"Best Model Saved Successfully!\")\n",
    "    else:\n",
    "        print(\"No models met the criteria for saving.\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    # Call the function if the module is executed as a script\n",
    "    train_and_save_models()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6955b332-bd38-471e-820e-0a130e7279fa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
