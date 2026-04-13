import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
from matplotlib.container import BarContainer

# ==========================================
# 1. Real-world Geographical Coordinates and Physical Distance Calculation
# ==========================================
# Updated to the most precise coordinates for the Tai Po Market base station
BASE_STATION = (22.444510, 114.170447)

# The complete 15 real-world delivery node coordinates
DELIVERY_NODES = [
    (22.434244, 114.188128),  # Dest 1: Ti Tao Tsuen House 15
    (22.436428, 114.182034),  # Dest 2: Savanna Garden Block 42
    (22.439245, 114.182323),  # Dest 3: 4138 Tai Po Road - Tai Po Kau
    (22.442473, 114.176151),  # Dest 4: Wong Yi Au Village Public Toilet
    (22.434132, 114.167191),  # Dest 5: 88 Organic Farm
    (22.438541, 114.165251),  # Dest 6: Wah's Store Cainiao Pick-up Point
    (22.441623, 114.160753),  # Dest 7: SF Smart Locker (Grand Dynasty View)
    (22.444682, 114.166511),  # Dest 8: Tai Po Pan Chung San Tsuen Pick-up Point P37
    (22.447275, 114.166360),  # Dest 9: Po Heung Estate Po Shun House Pick-up Point
    (22.451656, 114.161194),  # Dest 10: Cainiao Pick-up Point P34 Tai Wo Market
    (22.448902, 114.169665),  # Dest 11: P23
    (
        22.449547,
        114.176982,
    ),  # Dest 12: Tai Po Kwong Fuk Market Cainiao Pick-up Point P28
    (22.454516, 114.176513),  # Dest 13: SF Express Tai Po Fu Shin Shopping Centre
    (22.455899, 114.169003),  # Dest 14: Tai Yuen Commercial Centre LK01 SF Smart Locker
    (22.456571, 114.188370),  # Dest 15: Dai Kwai Street, Industrial Estate
]


def calculate_haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth surface.
    """
    R = 6371.0  # Earth's radius in kilometers
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


# Pre-calculate the real-world distance array for the 15 delivery nodes
DISTANCES_KM = np.array(
    [calculate_haversine(BASE_STATION, node) for node in DELIVERY_NODES]
)

# Match 3.20code.py parcel destination logic:
# 30%-40% parcels go to special points (10, 11, 14), remaining parcels are
# uniformly distributed among the other 12 destinations.
SPECIAL_POINTS = [10, 11, 14]
SPECIAL_RATIO_RANGE = (0.30, 0.40)


def build_destination_probabilities():
    """Build destination probabilities aligned with the 3.20code destination weighting logic.

    Special points (10, 11, 14) jointly receive 30%-40% demand (modeled by
    the midpoint 35%), while the remaining demand is evenly distributed across
    the other 12 destinations.
    """
    n_dest = len(DELIVERY_NODES)
    probs = np.zeros(n_dest, dtype=float)

    special_ratio = np.mean(SPECIAL_RATIO_RANGE)
    regular_points = [i for i in range(1, n_dest + 1) if i not in SPECIAL_POINTS]

    # Split special share equally among special points, regular share equally among regular points.
    for p in SPECIAL_POINTS:
        probs[p - 1] = special_ratio / len(SPECIAL_POINTS)
    for p in regular_points:
        probs[p - 1] = (1.0 - special_ratio) / len(regular_points)

    return probs


# ==========================================
# 2. Define Statistical Tools (Mean and Variance Calculation)
# ==========================================
def uniform_stats(min_val, max_val):
    """Calculate the Expected Value (Mean) and Variance of a Uniform Distribution U(a,b)"""
    mean = (min_val + max_val) / 2.0
    variance = ((max_val - min_val) ** 2) / 12.0
    return mean, variance


def normal_stats(mean, std_dev):
    """Return the Expected Value (Mean) and Variance of a Normal Distribution"""
    return mean, std_dev**2


# ==========================================
# 3. Core Analytical Engine
# ==========================================
STATIONS_DB = {
    "Admiralty": {"mtr": 29, "truck": (35, 75)},
    "Exhibition Ctr": {"mtr": 27, "truck": (35, 70)},
    "Hung Hom": {"mtr": 22, "truck": (25, 55)},
    "Mong Kok East": {"mtr": 18, "truck": (20, 45)},
    "Kowloon Tong": {"mtr": 15, "truck": (20, 45)},
    "Tai Wai": {"mtr": 11, "truck": (15, 30)},
    "Sha Tin": {"mtr": 8, "truck": (12, 25)},
    "Fo Tan": {"mtr": 5, "truck": (10, 20)},
    "University": {"mtr": 3, "truck": (8, 15)},
}

results_data = []

# --- Pre-calculate Last-Mile Mean and Variance (non-uniform destination probabilities) ---

DESTINATION_PROBS = build_destination_probabilities()

# Truck Last-Mile: Distance * 1.5 (tortuosity) / 0.5 km/min (30km/h) + U(5,10) Parking/Delivery penalty
truck_drive_times = (DISTANCES_KM * 1.5) / 0.5
truck_drive_mean = np.sum(truck_drive_times * DESTINATION_PROBS)
truck_drive_var = np.sum(
    DESTINATION_PROBS * (truck_drive_times - truck_drive_mean) ** 2
)  # Variance of weighted discrete spatial distribution
park_mean, park_var = uniform_stats(5, 10)
truck_lastmile_mean = truck_drive_mean + park_mean
truck_lastmile_var = truck_drive_var + park_var

# Drone Last-Mile: Distance / 0.9 km/min (54km/h) + 1 minute drop-off time
drone_flight_times = (DISTANCES_KM / 0.9) + 1.0
drone_flight_mean = np.sum(drone_flight_times * DESTINATION_PROBS)
drone_flight_var = np.sum(
    DESTINATION_PROBS * (drone_flight_times - drone_flight_mean) ** 2
)

# -----------------------------------------------------------

for station, data in STATIONS_DB.items():

    # === MODEL A: Traditional Truck Analytical Calculation ===
    m_load, v_load = uniform_stats(5, 10)
    m_road, v_road = uniform_stats(data["truck"][0], data["truck"][1])
    m_unload, v_unload = uniform_stats(5, 10)

    truck_total_mean = m_load + m_road + m_unload + truck_lastmile_mean
    truck_total_var = v_load + v_road + v_unload + truck_lastmile_var
    truck_total_std = math.sqrt(truck_total_var)
    truck_p95 = truck_total_mean + (
        1.645 * truck_total_std
    )  # Normal approximation based on Central Limit Theorem (CLT)

    results_data.append(
        {
            "Station": station,
            "Mode": "Traditional Truck",
            "Average_Time": truck_total_mean,
            "P95_Time": truck_p95,
            "Std_Dev": truck_total_std,
        }
    )

    # === MODEL B: MTR + Drone Analytical Calculation ===
    m_first, v_first = uniform_stats(5, 10)
    m_wait, v_wait = uniform_stats(0, 6)
    m_rail, v_rail = normal_stats(
        data["mtr"], 0.5
    )  # MTR standard deviation set to a highly stable 0.5 minutes
    m_transfer, v_transfer = uniform_stats(3, 5)
    m_drone_load, v_drone_load = uniform_stats(2, 4)

    mtr_total_mean = (
        m_first + m_wait + m_rail + m_transfer + m_drone_load + drone_flight_mean
    )
    mtr_total_var = (
        v_first + v_wait + v_rail + v_transfer + v_drone_load + drone_flight_var
    )
    mtr_total_std = math.sqrt(mtr_total_var)
    mtr_p95 = mtr_total_mean + (1.645 * mtr_total_std)

    results_data.append(
        {
            "Station": station,
            "Mode": "MTR + Drone",
            "Average_Time": mtr_total_mean,
            "P95_Time": mtr_p95,
            "Std_Dev": mtr_total_std,
        }
    )

df_results = pd.DataFrame(results_data)

# ==========================================
# 4. Visualization and Chart Generation
# ==========================================
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))

chart = sns.barplot(
    data=df_results,
    x="Station",
    y="Average_Time",
    hue="Mode",
    palette=["#d62728", "#1f77b4"],
    alpha=0.9,
    edgecolor="black",
)

for container in chart.containers:
    if isinstance(container, BarContainer):
        chart.bar_label(
            container, fmt="%.1f", padding=3, fontsize=11, fontweight="bold"
        )

plt.title(
    "Analytical Calculation: Traditional Truck vs. MTR + Drone\n(Impact of Origin Distance on Expected Delivery Time)",
    fontsize=15,
    fontweight="bold",
    pad=20,
)
plt.ylabel("Expected Total Time (Minutes)", fontsize=13, fontweight="bold")
plt.xlabel("Origin Station (East Rail Line)", fontsize=13, fontweight="bold")
plt.xticks(fontsize=11)
plt.ylim(0, 140)
plt.legend(title="Transport Mode", loc="upper right", frameon=True)
plt.tight_layout()
plt.show()

# Print precise comparison data
comparison = df_results.pivot(index="Station", columns="Mode", values="Average_Time")
comparison["Time Saved (min)"] = (
    comparison["Traditional Truck"] - comparison["MTR + Drone"]
)
comparison["Std_Dev_Truck"] = df_results[
    df_results["Mode"] == "Traditional Truck"
].set_index("Station")["Std_Dev"]
comparison["Std_Dev_MTR"] = df_results[df_results["Mode"] == "MTR + Drone"].set_index(
    "Station"
)["Std_Dev"]
print(
    comparison[
        [
            "Traditional Truck",
            "MTR + Drone",
            "Time Saved (min)",
            "Std_Dev_Truck",
            "Std_Dev_MTR",
        ]
    ].round(1)
)


# ==========================================
# 5. Certainty Premium Proof Chart
# ==========================================
station_order = list(STATIONS_DB.keys())

truck_rows = (
    df_results[df_results["Mode"] == "Traditional Truck"]
    .set_index("Station")
    .reindex(station_order)
)
mtr_rows = (
    df_results[df_results["Mode"] == "MTR + Drone"]
    .set_index("Station")
    .reindex(station_order)
)

x = np.arange(len(station_order))

truck_mean = truck_rows["Average_Time"].to_numpy(dtype=float)
truck_std = truck_rows["Std_Dev"].to_numpy(dtype=float)
mtr_mean = mtr_rows["Average_Time"].to_numpy(dtype=float)
mtr_std = mtr_rows["Std_Dev"].to_numpy(dtype=float)

# Under normal approximation, P10/P90 uses z ~= 1.2816.
z_10_90 = 1.2816
truck_p10 = truck_mean - z_10_90 * truck_std
truck_p90 = truck_mean + z_10_90 * truck_std
mtr_p10 = mtr_mean - z_10_90 * mtr_std
mtr_p90 = mtr_mean + z_10_90 * mtr_std

truck_band = truck_p90 - truck_p10
mtr_band = mtr_p90 - mtr_p10
certainty_reduction = (1 - (np.mean(mtr_band) / np.mean(truck_band))) * 100

fig, ax = plt.subplots(figsize=(13, 7))

ax.fill_between(
    x,
    truck_p10,
    truck_p90,
    color="#d62728",
    alpha=0.18,
    label="Traditional Truck Uncertainty Band (P10-P90)",
)
ax.fill_between(
    x,
    mtr_p10,
    mtr_p90,
    color="#1f77b4",
    alpha=0.20,
    label="MTR + Drone Uncertainty Band (P10-P90)",
)

ax.plot(
    x,
    truck_mean,
    color="#d62728",
    linewidth=2.8,
    marker="o",
    markersize=6,
    label="Traditional Truck Mean Time",
)
ax.plot(
    x,
    mtr_mean,
    color="#1f77b4",
    linewidth=2.8,
    marker="o",
    markersize=6,
    label="MTR + Drone Mean Time",
)

ax.set_title(
    "Certainty Premium Proof: Variability Compression in Delivery Time",
    fontsize=15,
    fontweight="bold",
    pad=16,
)
ax.set_xlabel("Origin Station (East Rail Line)", fontsize=12, fontweight="bold")
ax.set_ylabel("End-to-End Delivery Time (Minutes)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(station_order, rotation=15, ha="right")
ax.grid(True, axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper left", fontsize=10, frameon=True)

ax.text(
    0.99,
    0.03,
    (
        f"Average uncertainty band reduction: {certainty_reduction:.1f}%\n"
        f"(Traditional Truck: {np.mean(truck_band):.1f} min, "
        f"MTR + Drone: {np.mean(mtr_band):.1f} min)"
    ),
    transform=ax.transAxes,
    fontsize=10.5,
    ha="right",
    va="bottom",
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
)

plt.tight_layout()

output_dir = os.path.join(os.path.dirname(__file__), "simulation_data")
os.makedirs(output_dir, exist_ok=True)
certainty_chart_path = os.path.join(output_dir, "certainty_premium_proof.png")
plt.savefig(certainty_chart_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nCertainty premium chart saved to: {certainty_chart_path}")
