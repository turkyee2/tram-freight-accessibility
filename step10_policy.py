"""
후처리 — 위험신호개수 산출 및 군집1(구조적 압박형) A/B 세부 유형 확정

준비: outputs/final_result.csv (pipeline_v3.py 실행 결과)
실행: python step10_policy.py
산출: outputs/final_result_v2.csv
"""

import pandas as pd

df = pd.read_csv("outputs/final_result.csv")

# ── 위험신호(참고변수 중 이진·이산형) 목록 ─────────────────────
# 이면도로 여부는 "대안 존재" 신호라 성격이 반대이므로 위험신호에서 제외
RISK_FLAGS = ["반경100m_단속카메라개수", "commercial_overlap"]
RISK_FLAGS = [c for c in RISK_FLAGS if c in df.columns]
print("위험신호 변수:", RISK_FLAGS)

# 각 신호를 0/1로 정규화(1 이상이면 1로 간주) 후 개수 합산 — 가중치 없는 단순 카운트
df["위험신호개수"] = df[RISK_FLAGS].apply(lambda col: (col > 0).astype(int)).sum(axis=1)

print("\n전체 위험신호개수 분포:")
print(df["위험신호개수"].value_counts().sort_index())

# ── 군집별 참고변수 평균 (사후 해석) ────────────────────────
ref_cols = [c for c in ["이면도로 여부", "반경내총면수", "반경100m_단속카메라개수",
                        "commercial_overlap", "위험신호개수"] if c in df.columns]
print("\n군집별 참고변수 평균:")
print(df.groupby("군집_계층적")[ref_cols].mean().round(3).to_string())

# ── 군집1(구조적 압박형) 내부를 상권중첩 여부로 A/B 세분화 ──────
if "commercial_overlap" in df.columns:
    c1 = df[df["군집_계층적"] == 1]
    print(f"\n군집1({len(c1)}개) 내 상권중첩 분포:")
    print(c1["commercial_overlap"].value_counts().to_string())

    def assign_subtype(row):
        if row["군집_계층적"] != 1:
            return None
        return "A(시간이동형)" if row["commercial_overlap"] == 1 else "B(공간이동형)"

    df["군집1_세부유형"] = df.apply(assign_subtype, axis=1)
    print("\n군집1 세부유형 분포:")
    print(df["군집1_세부유형"].value_counts(dropna=True).to_string())

# ── 최종 정책유형 컬럼 생성 (군집0/2는 그대로, 군집1만 세분화) ──
def final_policy(row):
    if row["군집_계층적"] == 2:
        return "C(구조적 제약형)"
    if row["군집_계층적"] == 0:
        return "저위험(상시관리형)"
    return row.get("군집1_세부유형", "A/B(추가확인필요)")

df["최종정책유형"] = df.apply(final_policy, axis=1)

print("\n최종 정책유형 분포:")
print(df["최종정책유형"].value_counts().to_string())

df.to_csv("outputs/final_result_v2.csv", index=False, encoding="utf-8-sig")
print("\n완료 -> outputs/final_result_v2.csv (최종정책유형 컬럼 포함, GIS 담당 전달용)")
