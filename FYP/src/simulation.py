import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ==========================================
# 1. Real-world Geographical Coordinates and Physical Distance Calculation
# ==========================================
BASE_STATION = (22.4439, 114.1644)  # Tai Po Market MTR Station

DELIVERY_NODES = [
    (22.4300, 114.1850),  # Tai Po Kau San Wai
    (22.4285, 114.1682),  # 192 Ban Shan Chau
    (22.4402, 114.1568),  # Ma Wo
    (22.4610, 114.1735),  # Ha Hang
    (22.4583, 114.1820),  # Tai Po Industrial Estate
    (22.4580, 114.1710),  # Fu Heng Estate
    (22.4465, 114.1725),  # Kwong Fuk Estate
    (22.4530, 114.1680),  # Tai Po Centre
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


# Pre-calculate the real-world distance array for the 8 delivery nodes
DISTANCES_KM = np.array(
    [calculate_haversine(BASE_STATION, node) for node in DELIVERY_NODES]
)


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

# --- Pre-calculate Last-Mile Mean and Variance (Assuming equal probability for all 8 nodes) ---

# Truck Last-Mile: Distance * 1.5 (tortuosity) / 0.5 km/min (30km/h) + U(5,10) Parking/Delivery penalty
truck_drive_times = (DISTANCES_KM * 1.5) / 0.5
truck_drive_mean = np.mean(truck_drive_times)
truck_drive_var = np.var(
    truck_drive_times
)  # Variance of the discrete spatial distribution
park_mean, park_var = uniform_stats(5, 10)
truck_lastmile_mean = truck_drive_mean + park_mean
truck_lastmile_var = truck_drive_var + park_var

# Drone Last-Mile: Distance / 0.9 km/min (54km/h) + 1 minute drop-off time
drone_flight_times = (DISTANCES_KM / 0.9) + 1.0
drone_flight_mean = np.mean(drone_flight_times)
drone_flight_var = np.var(drone_flight_times)

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
    chart.bar_label(container, fmt="%.1f", padding=3, fontsize=11, fontweight="bold")

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
