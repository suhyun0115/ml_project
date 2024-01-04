# data_preprocessing_module.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    df_adu = pd.read_csv('samadult.csv')
    df_family = pd.read_csv('familyxx.csv')
    df_household = pd.read_csv('househld.csv', index_col='HHX')

    merged_df = pd.merge(df_adu, df_family, how='inner', on=['HHX', 'FMX'])
    df = pd.merge(df_household, merged_df, how='inner', on='HHX')

    df_col = pd.read_excel('분석변수_총정리.xls')
    col = df_col['Unnamed: 0'].tolist()
    col = [i.replace(' ', '') for i in col]
    df = df[col]

    col_list = df_col['Unnamed: 2'].tolist()
    df.columns = col_list

    df = df[~df['당뇨유무'].isin([3, 7, 9])]
    df.fillna(0, inplace=True)
    
    col_exclude = ['체질량지수', '나이', '결혼여부']
    for column in df.columns:
        if column not in col_exclude:
            df.loc[df[column].isin([3, 7, 8, 9, 996, 997, 998, 999]), column] = 0
    
    df.loc[df['체질량지수'] >= 7000, '체질량지수'] = 0
    df.loc[df['체질량지수'] >= 4000, '당뇨유무'] = 1
    df['신체활동빈도'] = df['신체활동빈도'].apply(lambda x: round(x / 4) if 10 <= x < 50 else round(x / 48) if x >= 50 else x)
    df.loc[df['의료비지출에 대한 인식'].isin([1, 2]), '의료비지출에 대한 인식'] = 1
    df.loc[df['의료비지출에 대한 인식'].isin([3, 4]), '의료비지출에 대한 인식'] = 2
    df.loc[df['결혼여부'].isin([1, 2, 3]), '결혼여부'] = 1
    df.loc[df['결혼여부'].isin([4, 5, 6, 7, 8, 9]), '결혼여부'] = 2
    
    col_1 = ['체질량지수', '신체활동빈도', '나이', '응답한 가구 내 아동의 수는 몇 명?', '응답한 가구 내 사람의 수는 몇 명?']
    col_2 = ['고혈압유무', '현재 임신여부', '우울증 여부', '체중문제', '알콜 및 약물 남용',
            '수술 후유증', '신체노화', '피로무기력증', '흡연유무', '1년내 음주경험',
            '고지혈증없음', '보통 아플 때 가는 곳', '의료비지출에 대한 인식',
            '인터넷 건강정보 검색유무 1년내', 'sex', '결혼여부',
            '균형 잡힌 식사를 할 여유가 없었습니다']

    df_encoded = pd.get_dummies(df[col_2], columns=col_2)
    df_encoded[col_1] = df[col_1]

    return df_encoded

def create_false_dataframe(df_encoded):
    columns_list = ['체질량지수', '신체활동빈도', '나이', '응답한 가구 내 아동의 수는 몇 명?', '응답한 가구 내 사람의 수는 몇 명?',
                    '고혈압유무_yes', '고혈압유무_no', '현재 임신여부_yes', '현재 임신여부_no', '우울증 여부_yes',
                    '우울증 여부_no', '체중문제_yes', '체중문제_no', '알콜 및 약물 남용_yes', '알콜 및 약물 남용_no',
                    '수술 후유증_yes', '수술 후유증_no', '신체노화_yes', '신체노화_no', '피로무기력증_yes',
                    '피로무기력증_no', '흡연유무_yes', '흡연유무_no', '1년내 음주경험_yes', '1년내 음주경험_no',
                    '고지혈증없음_yes', '고지혈증없음_no', '보통 아플 때 가는 곳_yes', '보통 아플 때 가는 곳_no',
                    '의료비지출에 대한 인식_yes', '의료비지출에 대한 인식_no', '인터넷 건강정보 검색유무 1년내_yes',
                    '인터넷 건강정보 검색유무 1년내_no', 'sex_yes', 'sex_no', '결혼여부_yes', '결혼여부_no',
                    '균형 잡힌 식사를 할 여유가 없었습니다_yes', '균형 잡힌 식사를 할 여유가 없었습니다_no']

    df_false = pd.DataFrame(False, index=[0], columns=columns_list)

    for col in df_false.columns:
        if col not in df_encoded.columns:
            df_encoded[col] = df_false[col]

    df_encoded = df_encoded[columns_list].fillna(False)

    return df_encoded

def preprocess_and_save_csv(df_encoded, output_prefix):
    df_encoded['당뇨유무'] = df['당뇨유무']
    df_encoded = df_encoded.loc[:, ~df_encoded.columns.str.endswith('_0.0') & ~df_encoded.columns.str.endswith('_0')]
    df = df_encoded

    df.loc[df['당뇨유무'] == 2, '당뇨유무'] = 0
    df.columns = [col.replace('_1', '_yes').replace('_2', '_no').replace('_1.0', '_yes').replace('_2.0', '_no').replace('.0', '') for col in df.columns]

    df_1 = df[df['당뇨유무'] == 1]
    df_2 = df[df['당뇨유무'] == 0]

    df_2_1 = df_2.iloc[0:3786]
    df_2_2 = df_2.iloc[3786:7572]
    df_2_3 = df_2.iloc[7572:11358]
    df_2_4 = df_2.iloc[11358:15144]
    df_2_5 = df_2.iloc[15144:18930]
    df_2_6 = df_2.iloc[18930:20869]

    df_01 = pd.concat([df_1, df_2_1], axis=0)
    df_02 = pd.concat([df_1, df_2_2], axis=0)
    df_03 = pd.concat([df_1, df_2_3], axis=0)
    df_04 = pd.concat([df_1, df_2_4], axis=0)
    df_05 = pd.concat([df_1, df_2_5], axis=0)
    df_06 = pd.concat([df_1, df_2_6], axis=0)

    dfs = [df_01, df_02, df_03, df_04, df_05, df_06]

    for idx, df in enumerate(dfs, start=1):
        print(f"Processing DataFrame {idx}")

        X = df.drop('당뇨유무', axis=1)
        y = df['당뇨유무']

        cols_without_no_yes = [col for col in X.columns if '_no' not in col and '_yes' not in col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train[cols_without_no_yes] = scaler.fit_transform(X_train[cols_without_no_yes])
        X_test[cols_without_no_yes] = scaler.transform(X_test[cols_without_no_yes])

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv(f'{output_prefix}_train_data_{idx}.csv', index=False)
        test_data.to_csv(f'{output_prefix}_test_data_{idx}.csv', index=False)
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == "__main__":
    df_encoded = load_data()
    df_encoded_false = create_false_dataframe(df_encoded)
    preprocess_and_save_csv(df_encoded_false, output_prefix="output")
    
    
    
# # 사용법
# from preprocessing_out import preprocess_and_save_csv

# # Replace 'your_input_data.csv' with the actual name of your CSV file
# df_encoded = load_data()
# df_encoded_false = create_false_dataframe(df_encoded)

# # Replace 'output' with the desired output prefix
# preprocess_and_save_csv(df_encoded_false, output_prefix="output")
