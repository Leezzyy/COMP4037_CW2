import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据（卡路里已调整版）
df_raw = pd.read_csv("Results_21MAR2022.csv")

ENV_COLS = [
    "mean_ghgs", "mean_land", "mean_watscar", "mean_eut",
    "mean_ghgs_ch4", "mean_ghgs_n2o", "mean_bio",
    "mean_watuse", "mean_acid"
]

# 分组均值
df_diet_agg = df_raw.groupby("diet_group", as_index=False)[ENV_COLS].mean()

# min-max 带 offset 归一化，防止 vegan 全 0
def min_max_scale_with_offset(series, offset=0.05):
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series * 0 + offset
    return (series - min_val) / (max_val - min_val) * (1 - offset) + offset

df_diet_scaled = df_diet_agg.copy()
for col in ENV_COLS:
    df_diet_scaled[col] = min_max_scale_with_offset(df_diet_scaled[col])

# 画雷达图
labels = ENV_COLS
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 每组画线
for i, row in df_diet_scaled.iterrows():
    values = row[labels].tolist()
    values += values[:1]
    ax.plot(angles, values, label=row["diet_group"], linewidth=1.5)
    ax.fill(angles, values, alpha=0.1)

# 坐标轴设置
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)

# 标题和图例
ax.set_title("Environmental Impact by Diet Group (Normalized)", pad=30, fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10)

# 范围设定
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("radar_chart_cleaned.png", dpi=300)
plt.show()
