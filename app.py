import streamlit as st
import pandas as pd
import openai
from collections import defaultdict

# 🎯 Streamlit 웹 앱 설정
st.set_page_config(page_title="특허 분류", layout="wide")

# 🔑 OpenAI API 키 입력
st.title("📂 LLM 기반 특허명(=발명명칭) 표준산업기술분류")
api_key = st.text_input("🔑 OPENAI API 키를 입력하세요", type="password")

# 📘 분류 체계 입력
st.subheader("📘 분류 체계 입력")
st.markdown("각 줄에 `대분류;중분류;소분류1,소분류2` 형식으로 입력하세요.")

default_example = """정보통신;소프트웨어;인공지능,딥러닝
정보통신;네트워크;5G,Wi-Fi
기계;자동차;전기차,자율주행"""

hierarchy_text = st.text_area("분류 체계 입력", value=default_example)

# 📂 CSV 파일 업로드 
uploaded_file = st.file_uploader("📂 발명명칭 CSV 파일을 업로드하세요(하나 이상의 열에 발명명칭 포함 가능)", type="csv")

# 🧠 분류 체계 파싱 함수
def parse_hierarchy(text):
    hierarchy = defaultdict(lambda: defaultdict(set))
    for line in text.strip().split("\n"):
        parts = line.strip().split(";")
        if len(parts) == 3:
            major, mid, subs = parts
            for sub in subs.split(","):
                hierarchy[major.strip()][mid.strip()].add(sub.strip())
    return {k: dict(v) for k, v in hierarchy.items()}

# 🔍 LLM 분류 함수들
def classify_major(text, hierarchy, client):
    majors = list(hierarchy.keys())
    prompt = f"""
다음 발명 내용을 보고 아래 대분류 중 가장 적절한 하나를 선택하세요.  
해당 사항이 없으면 '비관련기술'을 출력하세요.

가능 목록: {', '.join(majors)}

발명 내용: {text}

출력: (대분류 하나 또는 비관련기술)
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.2
    )
    result = response.choices[0].message.content.strip()
    return result if result in majors else "비관련기술"

def classify_mid(text, major, hierarchy, client):
    if major == "비관련기술" or major not in hierarchy:
        return "비관련기술"
    mids = list(hierarchy[major].keys())
    prompt = f"""
다음 발명 내용은 '{major}' 대분류에 속합니다.  
가장 적절한 중분류 하나를 선택하세요.  
해당 사항이 없으면 '비관련기술'을 출력하세요.

가능 목록: {', '.join(mids)}

발명 내용: {text}

출력: (중분류 하나 또는 비관련기술)
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.2
    )
    result = response.choices[0].message.content.strip()
    return result if result in mids else "비관련기술"

def classify_sub(text, major, mid, hierarchy, client):
    if "비관련기술" in (major, mid):
        return "비관련기술"
    subs = list(hierarchy[major][mid])
    prompt = f"""
'{major}' 대분류, '{mid}' 중분류에 속한 기술 중에서  
다음 발명 내용에 가장 적절한 소분류 하나를 선택하세요.  
해당 사항이 없으면 '비관련기술'을 출력하세요.

가능 목록: {', '.join(subs)}

발명 내용: {text}

출력: (소분류 하나 또는 비관련기술)
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.2
    )
    result = response.choices[0].message.content.strip()
    return result if result in subs else "비관련기술"

# ✅ 분류 수행
if api_key and uploaded_file and hierarchy_text.strip():
    client = openai.OpenAI(api_key=api_key)
    hierarchy = parse_hierarchy(hierarchy_text)

    df = pd.read_csv(uploaded_file, encoding="utf-8")
    df = df.applymap(lambda x: x.replace("\n", " ").replace("\t", " ") if isinstance(x, str) else x)

    # 발명명칭 텍스트 추출
    def extract_text(row):
        return ' '.join(str(v) for v in row.values if isinstance(v, str))

    # 전체 분류
    def classify_row(row):
        text = extract_text(row)
        major = classify_major(text, hierarchy, client)
        mid = classify_mid(text, major, hierarchy, client)
        sub = classify_sub(text, major, mid, hierarchy, client)
        return pd.Series([major, mid, sub])

    st.write("📊 **업로드된 데이터**")
    st.dataframe(df, height=400, use_container_width=True)

    st.info("🧠 LLM을 사용해 분류를 수행 중입니다. 잠시만 기다려주세요...")

    df[['대분류', '중분류', '소분류']] = df.apply(classify_row, axis=1)

    st.success("✅ 분류 완료!")
    st.write("📊 **분류 결과 데이터**")
    st.dataframe(df, height=600, use_container_width=True)

    output_file = "classified_output.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    st.download_button("📥 결과 CSV 다운로드", data=open(output_file, "rb"), file_name="classified_output.csv")
