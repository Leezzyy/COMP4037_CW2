import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 读取 CSV 数据（确保文件路径正确）
df = pd.read_csv("Results_21MAR2022_nokcaladjust.csv")

# 筛选 vegan 群体
df_vegan = df[df["diet_group"] == "vegan"].copy()

# 创建性别 + 年龄组合字段
df_vegan["group"] = df_vegan["sex"] + "_" + df_vegan["age_group"]

# 选择环境指标列
ENV_COLS = [
    "mean_ghgs", "mean_land", "mean_watscar", "mean_eut",
    "mean_ghgs_ch4", "mean_ghgs_n2o", "mean_bio",
    "mean_watuse", "mean_acid"
]

# 按 group 聚合并计算平均值
df_grouped = df_vegan.groupby("group")[ENV_COLS].mean().reset_index()

# 对每一列做 min-max 归一化
scaler = MinMaxScaler()
df_scaled = df_grouped.copy()
df_scaled[ENV_COLS] = scaler.fit_transform(df_grouped[ENV_COLS])

# 数据转为长格式
df_melted = df_scaled.melt(id_vars="group", var_name="indicator", value_name="value")
heatmap_data = df_melted.pivot(index="group", columns="indicator", values="value")

# 开始绘图
plt.figure(figsize=(13, 7))
ax = sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", linewidths=0.5, linecolor='gray')
plt.title("Environmental Impact by Gender & Age Group (Vegan Only)", fontsize=14)
plt.xlabel("Environmental Indicator", fontsize=12)
plt.ylabel("Gender + Age Group", fontsize=12)
plt.xticks(rotation=30)

# 添加解释说明（图下注释）
plt.figtext(
    0.5, -0.15,
    "Note: Values are min-max normalized within each environmental indicator.\n"
    "0 = lowest impact group; 1 = highest impact group (within vegan population).",
    wrap=True, horizontalalignment='center', fontsize=10
)

plt.tight_layout()
plt.savefig("vegan_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
