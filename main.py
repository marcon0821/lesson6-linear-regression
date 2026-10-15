import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="서울 기온 예측기",
    page_icon="🌡️",
    layout="centered"
)

st.title("🌡️ 서울 연평균 기온 분석 및 예측기")
st.markdown("서울 기온 데이터를 활용하여 연도별 평균 기온 추세를 분석하고, 원하는 연도의 기온을 예측해보세요.")

# 데이터 로드 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding='utf-8')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 데이터 전처리
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
df['연도'] = df['날짜'].dt.year

# 1. 2025년까지 필터링
df_filtered = df[df['연도'] <= 2025]

# 2. 연도별 평균기온 및 관측일수(결측치가 아닌 행 수) 계산
annual = df_filtered.groupby('연도').agg(
    평균기온=('평균기온', 'mean'),
    관측일수=('평균기온', 'count')
).reset_index()

# 3. 관측일이 300일 미만인 해 및 결측치 제외
annual_valid = annual[(annual['관측일수'] >= 300) & (annual['평균기온'].notna())].copy()

# 요약 정보 추출
start_year = int(annual_valid['연도'].min())
end_year = int(annual_valid['연도'].max())
num_years = len(annual_valid)

# 화면에 조건에 맞는 정보 표시
st.info(f"📊 **분석 데이터 요약**\n"
        f"- **직선을 만든 해의 개수:** {num_years}개\n"
        f"- **시작 연도:** {start_year}년\n"
        f"- **끝 연도:** {end_year}년 (기준: 2025년까지, 관측일 300일 이상)")

# 회귀 분석 및 상관계수 계산
x = annual_valid['연도'].values
y = annual_valid['평균기온'].values

corr_coef = np.corrcoef(x, y)[0, 1]
slope, intercept = np.polyfit(x, y, 1)

# 회귀선 예측값 생성
annual_valid['회귀예측기온'] = slope * annual_valid['연도'] + intercept

col1, col2 = st.columns(2)
with col1:
    st.metric(label="연도와 평균기온 상관계수 (r)", value=f"{corr_coef:.4f}")
with col2:
    st.metric(label="회귀식 기울기", value=f"{slope:.4f} °C/년")

st.markdown("---")

# Plotly 그래프 생성 (산점도 + 회귀 직선)
fig = px.scatter(
    annual_valid, 
    x='연도', 
    y='평균기온', 
    title="서울 연도별 평균기온 및 회귀 직선",
    labels={'연도': '연도', '평균기온': '평균기온 (°C)'},
    template='plotly_white'
)

fig.add_trace(
    go.Scatter(
        x=annual_valid['연도'],
        y=annual_valid['회귀예측기온'],
        mode='lines',
        name=f'회귀선 (y = {slope:.2f}x + {intercept:.2f})',
        line=dict(color='red', width=2)
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 슬라이더를 통한 특정 연도 기온 예측
st.subheader("🔮 특정 연도 기온 예측하기")
selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
        <h3 style="margin: 0; color: #31333F;">{selected_year}년 예상 서울 평균기온</h3>
        <h1 style="margin: 10px 0 0 0; color: #ff4b4b; font-size: 3rem;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)
