import streamlit as st
import pandas as pd
import re
from datetime import datetime

# ----------------------------
# Helper functions
# ----------------------------

day_map = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}


def parse_time_str(time_str: str) -> datetime.time:
    return datetime.strptime(time_str, "%H:%M").time()


def extract_sessions(time_field: str):
    pattern = r"(월|화|수|목|금|토|일) \[[^\]]+\] (\d{2}:\d{2})~(\d{2}:\d{2})"
    sessions = []
    for day, start, end in re.findall(pattern, time_field):
        sessions.append((day_map[day], parse_time_str(start), parse_time_str(end)))
    return sessions


def is_conflict(sessions_a, sessions_b):
    for d1, s1, e1 in sessions_a:
        for d2, s2, e2 in sessions_b:
            if d1 != d2:
                continue
            if max(s1, s2) < min(e1, e2):
                return True
    return False


def bayesian_average(rating, n, global_mean, k: int = 10):
    if pd.isna(rating) or rating == 0 or pd.isna(n):
        return global_mean
    return (n / (n + k)) * rating + (k / (n + k)) * global_mean


def select_offering(group_df: pd.DataFrame, current_sessions: list, global_mean: float):
    if group_df.empty:
        return None, None

    df = group_df.copy()
    df["bayes"] = df.apply(
        lambda r: bayesian_average(r["강의평"], r["평가수"], global_mean), axis=1
    )
    df = df.sort_values("bayes", ascending=False)

    primary = None
    backup = None

    # Primary 선정 (비충돌 중 최고)
    for _, row in df.iterrows():
        sess = extract_sessions(row["강의시간"])
        if not is_conflict(current_sessions, sess):
            primary = row
            current_sessions.extend(sess)  # 세션 확정
            break

    # Backup 선정 (Primary 제외하고 비충돌 중 최고)
    for _, row in df.iterrows():
        if primary is not None and row["과목코드"] == primary["과목코드"]:
            continue
        sess = extract_sessions(row["강의시간"])
        if not is_conflict(current_sessions, sess):
            backup = row
            break

    # 여전히 없으면, 충돌 무시하고 최고 평점 분반 사용
    if backup is None:
        if primary is None and not df.empty:
            backup = df.iloc[0]
        else:
            remain = df[df["과목코드"] != (primary["과목코드"] if primary is not None else None)]
            if not remain.empty:
                backup = remain.iloc[0]

    return primary, backup


def build_schedule(df: pd.DataFrame, selected_majors: list, num_general: int):
    primaries = []
    rec_records = []
    current_sessions: list = []

    global_mean = df["강의평"].replace(0, pd.NA).dropna().mean()

    def append_record(category: str, subj: str, primary_row, backup_row):
        rec_records.append(
            {
                "구분": category,
                "과목명": subj,
                "Primary 과목코드": primary_row["과목코드"] if primary_row is not None else "",
                "Primary 강의시간": primary_row["강의시간"] if primary_row is not None else "",
                "Primary 교수": primary_row["교수"] if primary_row is not None else "",
                "Primary 강의평": primary_row["강의평"] if primary_row is not None else pd.NA,
                "Backup 과목코드": backup_row["과목코드"] if backup_row is not None else "",
                "Backup 강의시간": backup_row["강의시간"] if backup_row is not None else "",
                "Backup 교수": backup_row["교수"] if backup_row is not None else "",
                "Backup 강의평": backup_row["강의평"] if backup_row is not None else pd.NA,
            }
        )

    # ---------- 전필 ----------
    for subj in df[df["구분"] == "전필"]["과목명"].unique():
        group = df[(df["구분"] == "전필") & (df["과목명"] == subj)]
        primary, backup = select_offering(group, current_sessions, global_mean)
        if primary is not None:
            primaries.append(primary)
        append_record("전필", subj, primary, backup)

    # ---------- 전선 ----------
    elective_df = df[df["구분"] == "전선"]
    for subj in selected_majors:
        group = elective_df[elective_df["과목명"] == subj]
        primary, backup = select_offering(group, current_sessions, global_mean)
        if primary is not None:
            primaries.append(primary)
        append_record("전선", subj, primary, backup)

    # ---------- 교선 ----------
    general_df = df[df["구분"] == "교선"].copy()
    general_df["bayes"] = general_df.apply(
        lambda r: bayesian_average(r["강의평"], r["평가수"], global_mean), axis=1
    )
    general_df = general_df.sort_values("bayes", ascending=False)

    added = 0
    for _, row in general_df.iterrows():
        if added >= num_general:
            break
        sess = extract_sessions(row["강의시간"])
        if is_conflict(current_sessions, sess):
            continue
        primary = row
        current_sessions.extend(sess)
        added += 1
        # Backup same subject(과목명) 내 다른 분반
        grp = general_df[(general_df["과목명"] == row["과목명"]) & (general_df["과목코드"] != row["과목코드"])]
        _, backup = select_offering(grp, current_sessions.copy(), global_mean)
        primaries.append(primary)
        append_record("교선", row["과목명"], primary, backup)

    primary_df = pd.DataFrame(primaries).reset_index(drop=True)
    rec_df = pd.DataFrame(rec_records).reset_index(drop=True)
    return primary_df, rec_df

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="강의 시간표 추천기", layout="wide")
st.title("강의 시간표 추천 프로그램")

uploaded_file = st.file_uploader("강의 데이터 파일을 업로드하세요 (탭 구분 텍스트)", type=["txt", "tsv"])

if uploaded_file is None:
    st.info("먼저 강의 데이터 파일을 업로드해주세요.")
    st.stop()

# 데이터 로드
try:
    df = pd.read_csv(uploaded_file, sep="\t")
except Exception as e:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

st.success("데이터 로드 완료!")

# --- Sidebar 옵션 ---

st.sidebar.header("옵션 설정")

# 전선 과목 목록
elective_options = df[df["구분"] == "전선"]["과목명"].unique().tolist()
selected_majors = st.sidebar.multiselect("전선 과목을 선택하세요", elective_options)

num_general = st.sidebar.radio("교선 과목 개수 선택", options=[1, 2], index=0, horizontal=True)

# 시간표 추천 버튼
if st.sidebar.button("시간표 추천 받기"):
    primary_df, rec_df = build_schedule(df, selected_majors, num_general)

    if primary_df.empty:
        st.warning("조건을 만족하는 시간표를 찾을 수 없습니다. 옵션을 조정해 보세요.")
        st.stop()

    st.subheader("추천 시간표")
    st.dataframe(primary_df[["과목코드", "과목명", "교수", "강의시간", "학점", "강의평", "평가수"]])

    st.subheader("과목별 대안")
    st.dataframe(
        rec_df[[
            "구분",
            "과목명",
            "Primary 과목코드",
            "Primary 강의시간",
            "Backup 과목코드",
            "Backup 강의시간",
        ]]
    )