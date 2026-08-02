"""
트램 화물차 접근성 유형 분류 — 분석 파이프라인 (v3)

변경사항 (v2 대비)
  - Gower 거리 도입: 이면도로 여부(이진) + 나머지 연속변수를 함께 군집화하되,
    이진변수가 유클리드 거리를 지배하는 문제를 구조적으로 방지
  - GMM 제거: Gower의 precomputed 거리와 GMM(연속 가우시안 가정)은 병행 불가
    → 계층적 군집화(average linkage) 단일 기법으로 진행
  - 반경내총면수·반경100m_단속카메라개수 재포함 (v2에서는 제외했었음)

실행: python pipeline.py
준비: master_table.csv (cp949 인코딩)
산출: outputs/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_SM = True
except ImportError:
    HAS_SM = False

# -- 한글 폰트 --------------------------------------------------
plt.rcParams["font.family"] = "NanumBarunGothic"   # Mac: "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

IN_PATH = "master_table.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================
# 컬럼 정의
# ==============================================================
ID_COLS = ["title", "name", "latitude", "longitude", "도로명", "최근접주차장ID"]

MODEL_COLS = [                       # 군집화 투입 (6개, 순수 연속변수만)
    "차로 수",
    "트램_차로감소율",
    "TTI_차이_상행",
    "TTI_차이_하행",
    "최근접거리(m)",
    "최근접주차장면수",
    # "상권_업종구성비",            # 확보되면 주석 해제
]

CAT_COLS = []                         # Gower 미사용 (사유는 하단 STEP 5 참고)

RESERVE_COLS = ["이면도로 여부", "반경내총면수", "반경100m_단속카메라개수",
                "반경내주차장개수", "commercial_overlap"]   # 참고용 (모델 미투입, 사후 해석·정책설계용)

NON_NEGATIVE = ["차로 수", "최근접거리(m)", "반경100m_단속카메라개수",
                "최근접주차장면수", "반경내주차장개수", "반경내총면수"]


# ==============================================================
# STEP 1. 로드 및 결측치
# ==============================================================
df = pd.read_csv(IN_PATH, encoding="cp949")
df.columns = df.columns.str.strip()
print(f"[1] 데이터: {df.shape[0]}행 x {df.shape[1]}열")

MODEL_COLS = [c for c in MODEL_COLS if c in df.columns]
print(f"    모델 투입 변수 {len(MODEL_COLS)}개: {MODEL_COLS}")

missing = df[MODEL_COLS].isnull().sum()
print("\n    결측치:")
print(missing[missing > 0] if missing.sum() else "    없음")

X = df[MODEL_COLS].copy()
if X.isnull().sum().sum():
    X = X.fillna(X.median(numeric_only=True))
    print("    -> 중앙값으로 대체 (서식3에 기재 필요)")

missing_reserve = [c for c in RESERVE_COLS if c not in df.columns]
if missing_reserve:
    print(f"    [!] RESERVE_COLS 중 master_table.csv에 없는 컬럼: {missing_reserve} -> 컬럼명 확인 필요")


# ==============================================================
# STEP 2. 이상치 확인 (제거 아님)
# ==============================================================
print("\n[2] 이상치 점검 (제거하지 않음)")
print(X.describe().T[["mean", "std", "min", "max"]].round(2))

for c in NON_NEGATIVE:
    if c in X.columns and (X[c] < 0).any():
        n = (X[c] < 0).sum()
        print(f"    [!] '{c}'에 음수 {n}건 - 입력 오류 여부 확인 필요")

plt.figure(figsize=(14, 6))
X.select_dtypes("number").boxplot(rot=45)
plt.title("변수별 분포 (이상치 확인용 · 제거하지 않음)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/boxplot.png", dpi=130)
plt.close()


# ==============================================================
# STEP 3. 상관행렬 (연속변수만)
# ==============================================================
cont_cols = [c for c in X.columns if c not in CAT_COLS]
corr = X[cont_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": .8})
plt.title("변수 간 상관행렬 (연속변수)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/corr_heatmap.png", dpi=130)
plt.close()

print("\n[3] 상관계수 |r| >= 0.8 (연속변수 기준):")
high = [(corr.index[i], corr.columns[j], round(corr.iloc[i, j], 3))
        for i in range(len(corr)) for j in range(i + 1, len(corr))
        if abs(corr.iloc[i, j]) >= 0.8]
for a, b, v in high:
    print(f"    {a} <-> {b} : {v}")
if not high:
    print("    없음")


# ==============================================================
# STEP 4. VIF (연속변수만)
# ==============================================================
if HAS_SM:
    Xv = X[cont_cols].assign(_const=1.0)
    vif = pd.DataFrame({
        "변수": cont_cols,
        "VIF": [variance_inflation_factor(Xv.values, i) for i in range(len(cont_cols))],
    }).sort_values("VIF", ascending=False)
    vif["판정"] = pd.cut(vif["VIF"], [0, 5, 10, np.inf], labels=["양호", "확인", "제거검토"])
    print("\n[4] VIF:")
    print(vif.to_string(index=False))
    vif.to_csv(f"{OUT_DIR}/vif.csv", index=False, encoding="utf-8-sig")
else:
    print("\n[4] VIF 건너뜀 (pip install statsmodels 후 재실행)")


# ==============================================================
# STEP 5. 표준화
# ==============================================================
# 참고: 이면도로 여부(이진)·반경내총면수·단속카메라개수(제로 편중)는
# 실측 검증 결과 유클리드/Gower 거리 모두에서 거리 계산을 지배해
# 나머지 연속변수의 변별력을 없애는 것으로 확인되어(88:21, 99:1:9 등
# 극단적 분할 발생) 군집화 입력에서 제외함. RESERVE_COLS로 보존하여
# STEP 8 해석 단계·정책 설계 단계에서 계속 활용함.
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print(f"\n[5] 표준화 완료 - 군집화 투입 변수 {X.shape[1]}개(순수 연속형, PCA 미적용)")


# ==============================================================
# STEP 6. 군집 수 결정
# ==============================================================
Z = linkage(Xs, method="ward")

plt.figure(figsize=(13, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90)
plt.title("계층적 군집화 덴드로그램 (Ward)")
plt.ylabel("병합 거리")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dendrogram.png", dpi=130)
plt.close()

sil = {}
for k in range(2, 7):
    lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
    sil[k] = silhouette_score(Xs, lab)

plt.figure(figsize=(6, 4))
plt.plot(list(sil.keys()), list(sil.values()), "o-")
plt.xlabel("군집 수 (k)"); plt.ylabel("실루엣 점수")
plt.title("군집 수별 실루엣 점수")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/silhouette.png", dpi=130)
plt.close()

BEST_K = max(sil, key=sil.get)
print(f"\n[6] 실루엣: { {k: round(v,3) for k,v in sil.items()} }")
print(f"    -> 최적 k = {BEST_K}")
print("    ※ 실루엣 1위와 차이가 작으면, 정책 해석 가능성을 고려해 K를 아래에서 직접 조정 가능")

K = 3   # <- 실루엣 결과 확인 후 직접 조정 (예: BEST_K 그대로 쓰려면 K = BEST_K)


# ==============================================================
# STEP 7. 군집화 실행 (계층적, Ward)
# ==============================================================
hier_lab = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Xs)
df["군집_계층적"] = hier_lab

print(f"\n[7] 군집화 완료 (k={K}, Ward linkage)")
print(pd.Series(hier_lab).value_counts().sort_index())


# ==============================================================
# STEP 8. 군집 해석
# ==============================================================
profile_cols = X.columns.tolist() + [c for c in RESERVE_COLS if c in df.columns]
profile = df.groupby("군집_계층적")[profile_cols].mean().round(3)
profile["포인트수"] = df.groupby("군집_계층적").size()
profile.to_csv(f"{OUT_DIR}/cluster_profile.csv", encoding="utf-8-sig")

print("\n[8] 군집별 변수 평균 (해석의 핵심 근거 · 참고변수 포함):")
print(profile.T.to_string())

prof_std = pd.DataFrame(Xs, columns=X.columns).groupby(hier_lab).mean()
plt.figure(figsize=(11, 5))
sns.heatmap(prof_std.T, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("군집별 변수 특성 (표준화 값 · 빨강=평균보다 높음)")
plt.xlabel("군집")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cluster_heatmap.png", dpi=130)
plt.close()


# ==============================================================
# STEP 9. 결과 저장
# ==============================================================
out_cols = ([c for c in ID_COLS if c in df.columns]
            + [c for c in X.columns if c in df.columns]
            + [c for c in RESERVE_COLS if c in df.columns]
            + ["군집_계층적"])
df[out_cols].to_csv(f"{OUT_DIR}/final_result.csv", index=False, encoding="utf-8-sig")

print(f"""
[완료] {OUT_DIR}/
   boxplot.png           이상치 확인
   corr_heatmap.png      상관행렬(연속변수)
   vif.csv               다중공선성(연속변수)
   dendrogram.png        군집 병합 과정 (Ward)
   silhouette.png        군집 수 근거
   cluster_profile.csv   군집별 평균  <- 해석·정책유형 부여용
   cluster_heatmap.png   군집 특성(연속변수, 시각화용)
   final_result.csv      포인트별 결과 <- GIS 담당 전달
""")
