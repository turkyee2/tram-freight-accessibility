"""
트램 화물차 접근성 유형 분류 — 분석 파이프라인 (v2)

변경사항 (v1 대비)
  - PCA 제거: 모델 투입 변수가 8개 수준이라 불필요하며, 원변수 유지가 해석에 유리
  - 이상치 "확인" 단계 추가: 제거하지 않고 데이터 오류만 점검
  - VIF 추가: 상관행렬(2변수)로 못 잡는 다중 변수 간 공선성 진단

실행: python pipeline.py
준비: data/master_table.csv
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
from sklearn.mixture import GaussianMixture
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
# 컬럼 정의 - 실제 CSV 헤더와 일치시킬 것
# ==============================================================
ID_COLS = ["title", "name", "latitude", "longitude", "도로명", "최근접주차장ID"]

MODEL_COLS = [                       # 군집화 투입 (8개)
    "차로 수",
    "트램_차로감소율",
    "이면도로 여부",
    "TTI_차이_상행",
    "TTI_차이_하행",
    "최근접거리(m)",
    "최근접주차장면수",
    "반경내총면수",
    "반경100m_단속카메라개수",
    # "상권_업종구성비",            # 확보되면 주석 해제
]

RESERVE_COLS = [                     # 정책설계 단계에서 사용 (모델 미투입)
    "최근접주차장면수", "반경내주차장개수", "반경내총면수",
]

# 음수가 나오면 안 되는 변수 (이상치 점검용)
NON_NEGATIVE = ["차로 수", "최근접거리(m)", "반경100m_단속카메라개수",
                "최근접주차장면수", "반경내주차장개수", "반경내총면수"]


# ==============================================================
# STEP 1. 로드 및 결측치
# ==============================================================
df = pd.read_csv(IN_PATH, encoding="cp949")
df.columns = df.columns.str.strip()          # 헤더 공백 제거
print(f"[1] 데이터: {df.shape[0]}행 x {df.shape[1]}열")

MODEL_COLS = [c for c in MODEL_COLS if c in df.columns]
print(f"    모델 투입 변수 {len(MODEL_COLS)}개: {MODEL_COLS}")

missing = df[MODEL_COLS].isnull().sum()
print("\n    결측치:")
print(missing[missing > 0] if missing.sum() else "    없음")

X = df[MODEL_COLS].copy()
if X.isnull().sum().sum():
    X = X.fillna(X.median())
    print("    -> 중앙값으로 대체 (서식3에 기재 필요)")


# ==============================================================
# STEP 2. 이상치 확인 (제거 아님 - 데이터 오류만 점검)
# ==============================================================
print("\n[2] 이상치 점검 (제거하지 않음)")
print(X.describe().T[["mean", "std", "min", "max"]].round(2))

for c in NON_NEGATIVE:
    if c in X.columns and (X[c] < 0).any():
        n = (X[c] < 0).sum()
        print(f"    [!] '{c}'에 음수 {n}건 - 입력 오류 여부 확인 필요")

print("\n    변수별 최대값 상위 3개 (실제 값인지 확인):")
for c in X.columns:
    top = X[c].nlargest(3)
    print(f"      {c}: {list(top.round(2).values)}")

plt.figure(figsize=(14, 6))
X.boxplot(rot=45)
plt.title("변수별 분포 (이상치 확인용 · 제거하지 않음)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/boxplot.png", dpi=130)
plt.close()


# ==============================================================
# STEP 3. 상관행렬
# ==============================================================
corr = X.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": .8})
plt.title("변수 간 상관행렬")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/corr_heatmap.png", dpi=130)
plt.close()

print("\n[3] 상관계수 |r| >= 0.8 (중복 정보 의심):")
high = [(corr.index[i], corr.columns[j], round(corr.iloc[i, j], 3))
        for i in range(len(corr)) for j in range(i + 1, len(corr))
        if abs(corr.iloc[i, j]) >= 0.8]
for a, b, v in high:
    print(f"    {a} <-> {b} : {v}")
if not high:
    print("    없음")

if "TTI_증감_상행" in X.columns and "TTI_증감_하행" in X.columns:
    r = abs(corr.loc["TTI_증감_상행", "TTI_증감_하행"])
    if r >= 0.8:
        X["TTI_증감_max"] = X[["TTI_증감_상행", "TTI_증감_하행"]].max(axis=1)
        X = X.drop(columns=["TTI_증감_상행", "TTI_증감_하행"])
        print(f"    -> TTI 상·하행 통합 (r={r:.3f}, max 채택)")


# ==============================================================
# STEP 4. VIF (다중공선성)
# ==============================================================
if HAS_SM:
    Xv = X.assign(_const=1.0)      # 절편 추가해야 VIF가 정확
    vif = pd.DataFrame({
        "변수": X.columns,
        "VIF": [variance_inflation_factor(Xv.values, i) for i in range(len(X.columns))],
    }).sort_values("VIF", ascending=False)
    vif["판정"] = pd.cut(vif["VIF"], [0, 5, 10, np.inf],
                        labels=["양호", "확인", "제거검토"])
    print("\n[4] VIF:")
    print(vif.to_string(index=False))
    vif.to_csv(f"{OUT_DIR}/vif.csv", index=False, encoding="utf-8-sig")

    drop = vif.loc[vif["VIF"] >= 10, "변수"].tolist()
    if drop:
        print(f"    [!] VIF 10 이상: {drop} - 제거 여부 판단 필요")
else:
    print("\n[4] VIF 건너뜀 (pip install statsmodels 후 재실행)")



# ==============================================================
# ── 극단적으로 치우친(0이 80% 이상) 변수는 군집화에서 제외 ──
# 로그변환으로도 해소되지 않는 사실상 이진(binary)에 가까운 변수:
# 이후 군집 해석 단계(cluster_profile)에서는 그대로 참고 변수로 사용
DROP_FROM_CLUSTERING = ["반경내총면수", "반경100m_단속카메라개수"]
X = X.drop(columns=[c for c in DROP_FROM_CLUSTERING if c in X.columns])
print("군집화에서 제외(사후 해석용으로 이관):", DROP_FROM_CLUSTERING)

# 로그변환 블록은 이제 필요 없으므로 제거하거나 그대로 둬도 무방
# (남은 변수 중 최근접거리(m), 최근접주차장면수는 치우침이 크지 않아 로그변환 없이도 무방)
# ==============================================================



# ==============================================================
# STEP 5. 표준화
# ==============================================================

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
print(f"\n[5] 표준화 완료 - 최종 투입 변수 {X.shape[1]}개 (PCA 미적용)")


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
print("    ※ 덴드로그램 확인 후 아래 K 값을 수동 조정 가능")

K = BEST_K          # <- 필요시 직접 수정 (예: K = 3)


# ==============================================================
# STEP 7. 군집화 (계층적 + GMM)
# ==============================================================
hier_lab = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Xs)

gmm = GaussianMixture(n_components=K, random_state=42, n_init=10).fit(Xs)
gmm_lab = gmm.predict(Xs)
conf = gmm.predict_proba(Xs).max(axis=1)

df["군집_계층적"] = hier_lab
df["군집_GMM"] = gmm_lab
df["소속확신도"] = conf.round(3)
df["경계구간"] = np.where(conf < 0.7, "경계(재검토)", "명확")

print(f"\n[7] 군집화 완료 (k={K})")
print("    두 기법 교차표:")
print(pd.crosstab(df["군집_계층적"], df["군집_GMM"]))
print(f"    경계구간(확신도<0.7): {(conf < 0.7).sum()}개 / {len(df)}개")


# ==============================================================
# STEP 8. 군집 해석 (원변수 그대로 -> 해석 용이)
# ==============================================================
profile = df.groupby("군집_계층적")[X.columns.tolist()].mean().round(3)
profile["포인트수"] = df.groupby("군집_계층적").size()
profile.to_csv(f"{OUT_DIR}/cluster_profile.csv", encoding="utf-8-sig")

print("\n[8] 군집별 변수 평균 (해석의 핵심 근거):")
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
            + ["군집_계층적", "군집_GMM", "소속확신도", "경계구간"])
df[out_cols].to_csv(f"{OUT_DIR}/final_result.csv", index=False, encoding="utf-8-sig")

print(f"""
[완료] {OUT_DIR}/
   boxplot.png           이상치 확인
   corr_heatmap.png      상관행렬
   vif.csv               다중공선성
   dendrogram.png        군집 병합 과정
   silhouette.png        군집 수 근거
   cluster_profile.csv   군집별 평균  <- 해석·정책유형 부여용
   cluster_heatmap.png   군집 특성
   final_result.csv      포인트별 결과 <- GIS 담당 전달
""")
