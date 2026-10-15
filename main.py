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
st.markdown("서울 기온 데이터를 활용하여 연도별 평균 기온 추세를 분석하고, 전체 기간과 최근 20년의 기온 상승 속도를 비교해보세요.")

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

# 전체 기간 회귀 분석
x_all = annual_valid['연도'].values
y_all = annual_valid['평균기온'].values

corr_coef = np.corrcoef(x_all, y_all)[0, 1]
slope_all, intercept_all = np.polyfit(x_all, y_all, 1)
per_century_all = slope_all * 100  # 100년당 기온 상승 폭

# 전체 기간 회귀예측기온 생성
annual_valid['회귀예측기온'] = slope_all * annual_valid['연도'] + intercept_all

# 최근 20년 기간 회귀 분석 (데이터 끝 연도 기준 최근 20개 연도)
recent_years_df = annual_valid[annual_valid['연도'] >= (end_year - 19)].copy()
x_rec = recent_years_df['연도'].values
y_rec = recent_years_df['평균기온'].values
slope_rec, intercept_rec = np.polyfit(x_rec, y_rec, 1)
per_century_rec = slope_rec * 100  # 최근 20년 기준 100년당 기온 상승 폭

st.markdown("---")
st.subheader("📈 기온 상승 속도 비교 (100년당 상승 기온)")

# 나란히 비교하여 크게 보여주기
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem; color: #555;">전체 기간 ({start_year}~{end_year}년)</p>
            <h2 style="margin: 5px 0 0 0; color: #1f77b4; font-size: 2.2rem;">+{per_century_all:.2f} °C</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: #777;">(1년당 +{slope_all:.4f} °C)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #fff1f0; text-align: center;">
            <p style="margin: 0; font-size: 0.9rem; color: #555;">최근 20년 ({int(recent_years_df['연도'].min())}~{end_year}년)</p>
            <h2 style="margin: 5px 0 0 0; color: #ff4b4b; font-size: 2.2rem;">+{per_century_rec:.2f} °C</h2>
            <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: #777;">(1년당 +{slope_rec:.4f} °C)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")
st.markdown(f"**전체 기간 상관계수 (r):** {corr_coef:.4f}")

st.markdown("---")

# Plotly 그래프 생성 (산점도 + 전체 회귀 직선)
fig = px.scatter(
    annual_valid, 
    x='연도', 
    y='평균기온', 
    title="서울 연도별 평균기온 및 전체 회귀 직선",
    labels={'연도': '연도', '평균기온': '평균기온 (°C)'},
    template='plotly_white'
)

fig.add_trace(
    go.Scatter(
        x=annual_valid['연도'],
        y=annual_valid['회귀예측기온'],
        mode='lines',
        name=f'전체 회귀선 (100년당 +{per_century_all:.2f}°C)',
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

predicted_temp = slope_all * selected_year + intercept_all

st.markdown(
    f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
        <h3 style="margin: 0; color: #31333F;">{selected_year}년 예상 서울 평균기온 (전체 회귀식 기준)</h3>
        <h1 style="margin: 10px 0 0 0; color: #ff4b4b; font-size: 3rem;">{predicted_temp:.2f} °C</h1>
    </div>
    """,
    unsafe_allow_html=True
)
