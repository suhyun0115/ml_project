# preprocessing_module.py

import pandas as pd
import joblib


def preprocess_and_normalize_save(df, output_file='datasets.csv', scaler_path='./scaler.pkl'):
    col_1 = ['체질량지수', '신체활동빈도', '나이', '응답한 가구 내 아동의 수는 몇 명?', '응답한 가구 내 사람의 수는 몇 명?']

    col_2 = ['고혈압유무', '현재 임신여부', '우울증 여부', '체중문제', '알콜 및 약물 남용',
                    '수술 후유증', '신체노화', '피로무기력증', '흡연유무', '1년내 음주경험',
                    '고지혈증없음', '보통 아플 때 가는 곳', '의료비지출에 대한 인식',
                    '인터넷 건강정보 검색유무 1년내', 'sex', '결혼여부',
                    '균형 잡힌 식사를 할 여유가 없었습니다']

    # 원-핫 인코딩 적용
    df_encoded = pd.get_dummies(df[col_2], columns=col_2)

    # col_1 열 추가
    df_encoded[col_1] = df[col_1]

    # 열 이름 정리
    df_encoded.columns = [col.replace('_1', '_yes').replace('_2', '_no').replace('_1.0', '_yes').replace('_2.0', '_no').replace('.0', '') for col in df_encoded.columns]

    # '_no', '_yes'가 없는 컬럼만 선택
    cols_without_no_yes = [col for col in df_encoded.columns if '_no' not in col and '_yes' not in col]
    df_encoded[cols_without_no_yes]

    # 정규화
    scaler = joblib.load(scaler_path)
    df_encoded[cols_without_no_yes] = scaler.transform(df_encoded[cols_without_no_yes])

    # Assuming df_encoded is your existing DataFrame and df_false is the DataFrame with all values set to False
    # List of columns
    columns_list = ['체질량지수', '신체활동빈도', '나이', '응답한 가구 내 아동의 수는 몇 명?', '응답한 가구 내 사람의 수는 몇 명?',
                    '고혈압유무_yes', '고혈압유무_no', '현재 임신여부_yes', '현재 임신여부_no', '우울증 여부_yes',
                    '우울증 여부_no', '체중문제_yes', '체중문제_no', '알콜 및 약물 남용_yes', '알콜 및 약물 남용_no',
                    '수술 후유증_yes', '수술 후유증_no', '신체노화_yes', '신체노화_no', '피로무기력증_yes',
                    '피로무기력증_no', '흡연유무_yes', '흡연유무_no', '1년내 음주경험_yes', '1년내 음주경험_no',
                    '고지혈증없음_yes', '고지혈증없음_no', '보통 아플 때 가는 곳_yes', '보통 아플 때 가는 곳_no',
                    '의료비지출에 대한 인식_yes', '의료비지출에 대한 인식_no', '인터넷 건강정보 검색유무 1년내_yes',
                    '인터넷 건강정보 검색유무 1년내_no', 'sex_yes', 'sex_no', '결혼여부_yes', '결혼여부_no',
                    '균형 잡힌 식사를 할 여유가 없었습니다_yes', '균형 잡힌 식사를 할 여유가 없었습니다_no']

    # Create a DataFrame with the given columns and set all values to False
    df_false = pd.DataFrame(False, index=[0], columns=columns_list)

    # Iterate through columns in df_false
    for col in df_false.columns:
        # Check if the column exists in df_encoded
        if col not in df_encoded.columns:
            # Update values in df_encoded with values from df_false
            df_encoded[col] = df_false[col]

    # Reassign columns to maintain the desired order
    df_encoded = df_encoded[columns_list].fillna(False)

    # Save to CSV file
    df_encoded.to_csv(output_file, index=False)
    
    return df_encoded

# 예시: 모듈 사용
# from preprocessing_module import preprocess_and_normalize_save
# df = pd.read_csv('your_dataset.csv')
# preprocess_and_normalize_save(df, output_file='datasets.csv')
