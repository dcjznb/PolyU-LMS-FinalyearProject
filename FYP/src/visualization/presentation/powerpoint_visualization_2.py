import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredText

# ==========================================
# 1. 核心参数准备 (Core Data Assumptions)
# ==========================================

# 传统货车参数
truck_params = {
    "Vehicle Cost (CapEx)": "350,000 HKD / Unit",
    "Driver Salary (Labor)": "26,054 HKD / Month\n(Census 2025)",
    "Diesel Price (Energy)": "29.27 HKD / Liter",
    "Fuel Consumption": "8.5L / 100km",
    "Parking & Tolls": "6,000 HKD / Month / Unit",
}

# 无人机参数
drone_params = {
    "Drone + Vertiport (CapEx)": "150,000 HKD / Unit\n(DJI FlyCart 30)",
    "Operator Salary (Labor)": "20,800 HKD / Month\n(1 Staff for 4 Drones)",
    "Electricity (Energy)": "1.106 HKD / kWh\n(Commercial Tariff)",
    "Battery Lifecycle": "200 Charges\n(3,183 HKD / Pair)",
    "MTR Space Share": "3,000 HKD / Month",
}

# ==========================================
# 2. 绘制参数对冲看板 (Parameter Dashboard)
# ==========================================

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis("off")  # 关闭坐标轴

# 背景颜色块区分
ax.add_patch(
    Rectangle((0, 0), 0.48, 1, facecolor="#fce4e4", alpha=0.5, transform=ax.transAxes)
)  # 浅红背景
ax.add_patch(
    Rectangle(
        (0.52, 0), 0.48, 1, facecolor="#e4f0fc", alpha=0.5, transform=ax.transAxes
    )
)  # 浅蓝背景

# 标题栏
ax.text(
    0.24,
    0.92,
    "Model A: Traditional Trucking\n[Core Parameters]",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="#8b0000",
    transform=ax.transAxes,
)
ax.text(
    0.76,
    0.92,
    "Model B: MTR + Drone\n[Core Parameters]",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="#08519c",
    transform=ax.transAxes,
)

# 绘制参数行
y_start = 0.75
y_step = 0.14
keys = list(truck_params.keys())

for i, key in enumerate(keys):
    y_pos = y_start - i * y_step

    # 画中间的分隔符和类别标签
    ax.text(
        0.5,
        y_pos,
        f"♦ {key.split('(')[0].strip()} ♦",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="gray",
        transform=ax.transAxes,
    )

    # 货车数据 (左侧)
    ax.text(
        0.24,
        y_pos,
        truck_params[key],
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#d73027",
        transform=ax.transAxes,
    )

    # 无人机数据 (右侧)
    drone_key = list(drone_params.keys())[i]
    ax.text(
        0.76,
        y_pos,
        drone_params[drone_key],
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#3182bd",
        transform=ax.transAxes,
    )

# 底部全局基准声明
plt.figtext(
    0.5,
    0.02,
    "Simulation Baseline: 10,500 Parcels / Month (420 Parcels / Day)",
    ha="center",
    fontsize=14,
    fontweight="bold",
    color="black",
    bbox={"facecolor": "#ffffcc", "alpha": 0.8, "pad": 5},
)

plt.title(
    "Fundamental Cost Drivers: The Architecture of OpEx",
    fontsize=22,
    fontweight="bold",
    pad=30,
)
plt.tight_layout()
plt.savefig("Cost_Parameters_Dashboard.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ 成功生成参数对照大屏：Cost_Parameters_Dashboard.png")
