import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe


# ==========================================
# 瀑布图绘制函数 (Waterfall Chart Function)
# ==========================================
def draw_waterfall(ax, categories, values, title, bar_color, total_color):
    """绘制单张瀑布图"""
    # 累加计算
    cumulative = np.cumsum([0] + values[:-1])
    y_span = max(cumulative[-1] + values[-1], 1)

    # 颜色设置 (最后一根总计柱子颜色不同)
    colors = [bar_color] * len(values)
    colors[-1] = total_color

    # 画柱子 (bottom参数实现瀑布悬浮效果)
    bars = ax.bar(
        categories, values, bottom=cumulative, color=colors, edgecolor="white"
    )

    # 添加连接线
    for i in range(1, len(values)):
        ax.plot(
            [i - 1 - 0.4, i + 0.4],
            [cumulative[i], cumulative[i]],
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    # 在柱子上添加具体金额数字
    for i, bar in enumerate(bars):
        yval = cumulative[i] + values[i] / 2  # 默认将文字放在柱子中间
        if i == len(values) - 1:  # 最后一根总计柱子，文字放顶上
            yval = cumulative[i] + values[i] + y_span * 0.012
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{int(values[i]):,} HKD\n(CPD: {values[i]/10500:.2f})",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.9,
                    "pad": 2,
                    "edgecolor": "none",
                },
            )
        else:
            # 小柱子标签改为放在柱子上方，避免数字挤在一起看不清
            is_small_bar = values[i] < y_span * 0.06
            if is_small_bar:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    cumulative[i] + values[i] + y_span * 0.006,
                    f"+{int(values[i]):,}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.9,
                        "pad": 1.5,
                        "edgecolor": "none",
                    },
                )
            else:
                text_obj = ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval,
                    f"+{int(values[i]):,}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )
                text_obj.set_path_effects(
                    [
                        pe.Stroke(linewidth=1.2, foreground="black", alpha=0.35),
                        pe.Normal(),
                    ]
                )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=15)


# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================

# 传统货车计算过程
truck_cats = [
    "Labor\n(4 Drivers)",
    "Fuel\n(850L)",
    "Tolls/Parking\n(4 Trucks)",
    "Total\nOpEx",
]
truck_vals = [104216, 24880, 24000, 153096]  # 最后一个必须等于前面的和

# 无人机计算过程
drone_cats = [
    "Labor\n(1 Operator)",
    "Battery\n(1050 Flights)",
    "Maintenance",
    "MTR/Energy\n(Share+Power)",
    "Total\nOpEx",
]
drone_vals = [
    20800,
    16695,
    5692,
    3580,
    46767,
]  # 为了图表不过于拥挤，把 MTR 和 电费合并为一项展示

# ==========================================
# 2. 绘制并排瀑布图 (Side-by-Side Waterfall)
# ==========================================

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6), sharey=True
)  # sharey=True 强制两者共用同一高度比例尺，形成视觉碾压

# 画左边的货车瀑布图
draw_waterfall(
    ax1,
    truck_cats,
    truck_vals,
    title="Model A: Trucking Cost Calculation",
    bar_color="#d73027",
    total_color="#8b0000",
)

# 画右边的无人机瀑布图
draw_waterfall(
    ax2,
    drone_cats,
    drone_vals,
    title="Model B: Drone Cost Calculation",
    bar_color="#3182bd",
    total_color="#08519c",
)

# 底部添加基准说明
plt.figtext(
    0.5,
    0.01,
    "Calculation Baseline: 10,500 Parcels / Month (Over 25 Operating Days)",
    ha="center",
    fontsize=12,
    fontweight="bold",
    bbox={"facecolor": "#ffffcc", "alpha": 0.5, "pad": 5},
)

plt.tight_layout(rect=(0, 0.05, 1, 1))  # 给底部的文字留出空间
plt.savefig(
    "Cost_Calculation_Waterfall.png", dpi=300, transparent=False, facecolor="white"
)
plt.show()

print("✅ 成功生成对冲瀑布图：Cost_Calculation_Waterfall.png")
