import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    col = df.columns.tolist()
    col = [i.replace(' ','') for i in col]
    df = df[col]

    # ... (전처리 코드 추가)

    return df

def main():
    df_adu = pd.read_csv('samadult.csv')
    df_family = pd.read_csv('familyxx.csv')
    df_household = pd.read_csv('househld.csv', index_col='HHX')

    # df_adu와 df_family를 HHX와 FMX 기준으로 inner join
    merged_df = pd.merge(df_adu, df_family, how='inner', on=['HHX', 'FMX'])

    # df_household와 merged_df를 HHX 기준으로 inner join
    df = pd.merge(df_household, merged_df, how='inner', on='HHX')

    df_col = pd.read_excel('분석변수_총정리.xls')
    col = df_col['Unnamed: 0'].tolist()
    col = [i.replace(' ','') for i in col]
    df = df[col]
    col_list = df_col['Unnamed: 2'].tolist()
    df.columns = col_list
    df = df[~df['당뇨유무'].isin([3, 7, 9])]
    df['응답한 가구 내 아동의 수는 몇 명?'] = df['응답한 가구 내 아동의 수는 몇 명?'].fillna(round(df_household.mean()))
    df['응답한 가구 내 사람의 수는 몇 명?'] = df['응답한 가구 내 사람의 수는 몇 명?'].fillna(round(df_household.mean()))
    df.fillna(0, inplace=True)
    col_exclude = ['체질량지수', '나이', '결혼여부']
    for column in df.columns:
        if column not in col_exclude:
            df.loc[df[column].isin([3, 7, 8, 9, 996,997,998,999]), column] = 0
    df.loc[df['체질량지수'] >= 7000, '체질량지수'] = 0
    df.loc[df['체질량지수'] >= 4000, '당뇨유무'] = 1
    df['신체활동빈도'] = df['신체활동빈도'].apply(lambda x: round(x / 4) if 10 <= x < 50 else round(x / 48) if x >= 50 else x)
    df.loc[df['의료비지출에 대한 인식'].isin([1,2]), '의료비지출에 대한 인식']= 1
    df.loc[df['의료비지출에 대한 인식'].isin([3,4]), '의료비지출에 대한 인식']= 2

    df.loc[df['결혼여부'].isin([1,2,3]), '결혼여부']= 1
    df.loc[df['결혼여부'].isin([4,5,6,7,8,9]), '결혼여부']= 2

    col_1 = ['체질량지수', '신체활동빈도', '나이', '응답한 가구 내 아동의 수는 몇 명?', '응답한 가구 내 사람의 수는 몇 명?']

    col_2 = ['고혈압유무', '현재 임신여부', '우울증 여부', '체중문제', '알콜 및 약물 남용',
                    '수술 후유증', '신체노화', '피로무기력증', '흡연유무', '1년내 음주경험',
                    '고지혈증없음', '보통 아플 때 가는 곳', '의료비지출에 대한 인식',
                    '인터넷 건강정보 검색유무 1년내', 'sex', '결혼여부',
                    '균형 잡힌 식사를 할 여유가 없었습니다']

    df_encoded = pd.get_dummies(df, columns=col_2)

    df_encoded[col_1] = df[col_1]

    df_encoded['당뇨유무'] = df['당뇨유무']
    df_encoded = df_encoded.loc[:, ~df_encoded.columns.str.endswith('_0.0') & ~df_encoded.columns.str.endswith('_0')]
    df = df_encoded
    df_1 = df[df['당뇨유무'] == 1]
    df_2 = df[df['당뇨유무'] == 2]

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

        # 레이블과 특성을 나누기
        X = df.drop('당뇨유무', axis=1)
        y = df['당뇨유무']

        # 학습 및 테스트 데이터 나누기
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 스케일링 적용
        scaler = joblib.load('scaler.pkl')
        X_train[col_1] = scaler.transform(X_train[col_1])
        X_test[col_1] = scaler.transform(X_test[col_1])

        # 데이터를 합쳐서 CSV 파일로 저장
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv(f'train_data_{idx}.csv', index=False)
        test_data.to_csv(f'test_data_{idx}.csv', index=False)

if __name__ == "__main__":
    preprocessing.main()
    

# 사용밥
# csv_file_path = 'your_input_data.csv'

# result_df = process_data(csv_file_path)
# print(result_df)
