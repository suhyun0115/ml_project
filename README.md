# Machine Learning Project
### 당뇨병 예측 머신러닝 분석 프로젝트 입니다.
---

___Team DBDBDeep 소개___


![조서현](https://github.com/seohyuny/ml_project/assets/154740829/299c4cb0-c98b-4358-b88d-c363000d818c)
![김유진](https://github.com/seohyuny/ml_project/assets/154740829/8dd8cbe7-5bec-4098-8d37-ed0dcf6e52e9)
![이수현](https://github.com/seohyuny/ml_project/assets/154740829/fd0e5ac6-50d6-49d6-96d0-d2240eb42a89)


조서현 / 김유진 / 이수현

---

___Concept___

[환경적 요인(생활습관) 당뇨 예측]

- ##### 참고문헌
  - [당뇨학회]<https://www.diabetes.or.kr/general/info/info_01.php?con=2>

- ##### 프로젝트에 사용된 분석 라이브러리
```
import pandas
import numpy
import sklearn
import streamlit
import joblib
```

--- 

___Process___

- ##### DataSets
  - [NHIS_2018] <https://www.cdc.gov/nchs/nhis/nhis_2018_data_release.htm>
  - Sample Adult file : samadult.csv

- ##### 분석 변수
[NHIS_2018](https://www.cdc.gov/nchs/nhis/nhis_2018_data_release.htm)

```
# 당뇨병 분석 변수 선정
import pandas as pd
df_a = pd.read_csv('samadult.csv')
df_a = df_a[['SEX','AGE_P','R_MARITL','DIBEV1','HYPEV','PREGNOW','DEP_2','AFLHCA18','BMI',
            'AFLHC29_','AFLHC31_','AFLHC32_','AFLHC33_','SMKEV','ALC1YR','CHLEV','VIGNO',
            'AUSUALPL','ASICNHC','HIT1A']]
```

![diabetes_age_count3](https://github.com/seohyuny/ml_project/assets/154740829/5b359a1c-bb3d-46e0-82bd-c98868b64571) ![cholesterol](https://github.com/seohyuny/ml_project/assets/154740829/84c3562f-3262-44bd-a775-1c0cd0ebbba2)


![diabetes_age_sex](https://github.com/seohyuny/ml_project/assets/154740829/1b8c6494-6fd8-42c6-ad52-510920ad11b3)




- ##### Data 전처리 (dataset 정보 및 가공)
```

```

- ##### Machine-Learning (Model 정보)
```

```




