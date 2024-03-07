import streamlit as st
import pandas as pd
import joblib
import numpy as np

#설문지 제목
st.markdown("""
     <h1 style='color: orange;'> 당뇨예측 설문조사 📑✨
     """, unsafe_allow_html=True)

# 기본 질문
st.markdown("<h2 style='color: gray; font-size: 24px;'>기본질문에 응답해주세요</h2>", unsafe_allow_html=True)

name = st.text_input("이름을 입력하세요:")
age = st.text_input("만 나이를 입력하세요:")
hights = st.text_input("키를 입력하세요(cm): ")
weights = st.text_input("체중을 입력하세요(kg): ")
activity= st.text_input('일주일에 몇 회의 신체활동(운동)을 하십니까?:주__회')

# 키와 체중이 모두 입력되었을 때만 BMI 계산
BMI = None  # BMI 변수 초기화      #체질량지수


# 키와 체중을 실수형으로 변환
if hights and weights:
    hights = float(hights)
    weights = float(weights)

    BMI = weights / ((hights / 100) ** 2)   # BMI 계산 (체중(kg) / 키(m)^2)

family = st.text_input("함께 거주하고 있는 가족 수는 몇 명입니까?: ")
child = st.text_input("가족 중 아동의 수는 몇 명입니까?: ")
sex = st.radio(
    "성별을 선택하세요:",
    options=[('남성'), ('여성')]
)

sex_str = "1" if sex == '남성' else "2" if sex == '여성' else "기타"

# 설문문항
st.markdown("<h3 style='color: gray;font-size: 24px;'>아래 질문에 대해 예(1) 또는 아니오(2)로 응답하여 주세요</h3>",
            unsafe_allow_html=True)
questions_and_choices = [
    ("고혈압 진단을 받은 적이 있으십니까?", ["예(1)", "아니오(2)"]),
    ("현재 임신 중이십니까?", ["예(1)", "아니오(2)"]),
    ("우울증 진단을 받은 적이 있으십니까?", ["예(1)", "아니오(2)"]),
    ("활동에 영향을 줄 정도의 과체중이십니까?", ["예(1)", "아니오(2)"]),
    ("알콜 중독이나 약물 중독 경험이 있으십니까?", ["예(1)", "아니오(2)"]),
    ("수술 후유증을 경험한 적이 있으십니까?", ["예(1)", "아니오(2)"]),
    ("신체 노화를 체감하십니까?", ["예(1)", "아니오(2)"]),
    ("쉽게 피로해지고 피곤을 잘 느끼시나요?", ["예(1)", "아니오(2)"]),
    ("흡연을 하십니까?", ["예(1)", "아니오(2)"]),
    ("음주를 하십니까?", ["예(1)", "아니오(2)"]),
    ("고지혈증 진단을 받은 적이 있으십니까?", ["예(1)", "아니오(2)"]),
    ("몸이 아플 때 병원이나 약국을 가는 편이십니까?", ["예(1)", "아니오(2)"]),
    ("의료비를 지출에 경제적인 부담을 느끼십니까?", ["예(1)", "아니오(2)"]),
    ("인터넷 등 의료 정보를 획득하기 위한 활동을 자주 하시나요?", ["예(1)", "아니오(2)"]),
    ("균형잡힌 식사를 하시는 편인가요?", ["예(1)", "아니오(2)"]),
    ("결혼여부", ["예(1)", "아니오(2)"])
]

# 응답 리스트로 저장
user_responses = []

# 설문 문항 응답
for question, choices in questions_and_choices:
    response = st.radio(question, choices)
    user_responses.append(response)


# 응답 데이터프레임 저장
summary_data = pd.DataFrame({
    "질문": [q for q, _ in questions_and_choices],
    "응답": user_responses
})

button_count = st.session_state.get('button_count', 0)  # 현재까지 버튼이 눌린 횟수를 가져옴

# 설문완료 버튼 클릭
if st.button('설문 완료', key='버튼1'):
    button_count += 1  # 버튼이 눌릴 때마다 횟수 증가
    st.session_state['button_count'] = button_count  # 현재 버튼 횟수 저장

# 홀수 번 누를 때 정보를 출력
if button_count % 2 == 1:
    st.markdown("<h2 style='color: blue; font-size: 22px;'>아래 건강정보를 확인하세요💡</h2>", unsafe_allow_html=True)
    st.markdown("----------------------------------------------------------------------------------------")
    st.markdown("<h2 style='color: blue; font-size: 18px;'>나의 기본정보</h2>", unsafe_allow_html=True)
    st.write("📛이름:", name)
    st.write("😺나이:", age)
    st.write("🎎성별:", sex_str)
    if BMI is not None:
        st.write("💪BMI:", round(BMI, 2))

    # 나의 건강 데이터 출력
    st.markdown("<h2 style='color: blue; font-size: 18px;'>나의 건강 데이터</h2>", unsafe_allow_html=True)
    # summary_data 데이터프레임 전치하여 새로운 데이터프레임 생성
    summary_data_1 = summary_data.transpose()

else:
    # 짝수 번 누를 때 정보를 숨김
    st.write("")

# st.markdown("----------------------------------------------------------------------------------------")


# 컬럼명 변경 및 "질문" 행 삭제
# summary_data_1 = summary_data.transpose()
column_names = summary_data['질문']
summary_data_2 = summary_data_1.iloc[:, :].rename(columns=column_names)
# st.write(summary_data_2)


# 성별, 나이, BMI, 아동수, 가족수 ,신체활동빈도 컬럼 추가
summary_data_2["sex"] = [sex] * len(summary_data_2)
summary_data_2["나이"] = [age] * len(summary_data_2)
summary_data_2["체질량지수"] = [BMI] * len(summary_data_2)
summary_data_2["신체활동빈도"] = [activity] * len(summary_data_2)
summary_data_2["응답한 가구 내 아동의 수는 몇 명?"] = [child] * len(summary_data_2)
summary_data_2["응답한 가구 내 사람의 수는 몇 명?"] = [family] * len(summary_data_2)



new_col=['고혈압유무','현재 임신여부','우울증 여부','체중문제','알콜 및 약물 남용',
         '수술 후유증','신체노화','피로무기력증','흡연유무','1년내 음주경험','고지혈증없음',
         '보통 아플 때 가는 곳','의료비지출에 대한 인식','인터넷 건강정보 검색유무 1년내',
         '균형 잡힌 식사를 할 여유가 없었습니다','결혼여부']


# 컬럼 이름 변경 및 결과를 원래의 데이터프레임에 할당
summary_data_3 = summary_data_2.rename(columns=dict(zip(summary_data_2.columns, new_col)))
summary_data_3 = summary_data_3.iloc[1:, :]


for column in summary_data_3.columns:
    summary_data_3[column] = summary_data_3[column].apply(lambda x: 1 if x == '예(1)' else (2 if x == '아니오(2)' else (1 if x == '남성' else (2 if x == '여성' else x))))

# st.write('##### summary_data_3 : (DF 가로 전치,성별,나이 컬럼명 변경, 응답 1,2 변환)')
# st.write(summary_data_3)


df = summary_data_3
df

# =====원핫인코딩======================
import streamlit as st
import pandas as pd
# import plotly.express as px


col_1 = ['체질량지수', '신체활동빈도', '나이', '응답한 가구 내 아동의 수는 몇 명?', '응답한 가구 내 사람의 수는 몇 명?']

col_2 = ['고혈압유무', '현재 임신여부', '우울증 여부', '체중문제', '알콜 및 약물 남용',
                '수술 후유증', '신체노화', '피로무기력증', '흡연유무', '1년내 음주경험',
                '고지혈증없음', '보통 아플 때 가는 곳', '의료비지출에 대한 인식',
                '인터넷 건강정보 검색유무 1년내', 'sex', '결혼여부',
                '균형 잡힌 식사를 할 여유가 없었습니다']

# 원-핫 인코딩 적용
df_encoded = pd.get_dummies(df[col_2], columns=col_2)
# df_encoded.columns

# col_1 열 추가
df_encoded[col_1] = df[col_1]
# df_encoded


# 열 이름 정리
df_encoded.columns = [col.replace('_1', '_yes').replace('_2', '_no').replace('_1.0', '_yes').replace('_2.0', '_no').replace('.0', '') for col in df_encoded.columns]
# df_encoded.columns


# '_no', '_yes'가 없는 컬럼만 선택
cols_without_no_yes = [col for col in df_encoded.columns if '_no' not in col and '_yes' not in col]
# df_encoded[cols_without_no_yes]
# st.write(df_encoded)

# 정규화
scaler = joblib.load('scaler.pkl')
df_encoded[cols_without_no_yes] = scaler.transform(df_encoded[cols_without_no_yes])

import pandas as pd

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
# st.write(df_encoded)

# 모델 로드및 예측
adaboost_model = joblib.load('./ada_model.pkl')
predictions = adaboost_model.predict(df_encoded)

# decision_function을 사용하여 확률 값 얻기
decision_values = adaboost_model.decision_function(df_encoded)

# 부호 있는 거리를 확률로 변환 (sigmoid 함수 사용)
# predictions는 예측된 클래스, probabilities는 해당 클래스에 속할 확률
if decision_values.ndim == 1:
    probabilities = 1 / (1 + np.exp(-decision_values))
else:
    probabilities = 1 / (1 + np.exp(-decision_values[:, 1]))

# 결과를 DataFrame으로 정리하여 확인
result_df = pd.DataFrame({'Prediction': predictions, 'Probability': probabilities})
# st.write(result_df)
# result_df.Probability


# 결과 예측!
# 확률을 백분율로 변환
result_df['Proba_1'] = result_df['Probability'] * 100

if (result_df['Prediction'] == 1).any():
    text = f'{name}님은 당뇨에 걸리지 않을 확률이 {result_df.loc[0, "Probability"] * 100:.2f}%입니다.'
else:
    text = f'{name}님은 당뇨에 걸릴 확률이 {result_df.loc[0, "Proba_1"]:.2f}%입니다.'

st.markdown(f"<span style='font-size: 28px; color: red;'>{text}</span>", unsafe_allow_html=True)


#=========한글 파일 설치===================================

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'gulim'
plt.rc('font', family='Malgun Gothic')
plt.rc('font', family='NanumGothic')


import warnings
# FutureWarning을 무시
warnings.simplefilter(action='ignore', category=FutureWarning)

# use_inf_as_na 경고 무시
pd.set_option('mode.use_inf_as_na', False)
#==========================================================

# 워드 클라우드 생성
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# 워드 클라우드 출력
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("당뇨 예측 결과")
st.pyplot(plt)


button_clicked = st.button('당뇨예방수칙 보러가기')

if button_clicked:
    st.write('당뇨 예방 수칙을 보러 가는 중')
    # 새로운 페이지로 이동하는 HTML 코드
    st.write('<meta http-equiv="refresh" content="0;URL=https://www.diabetes.or.kr/general/">', unsafe_allow_html=True)




