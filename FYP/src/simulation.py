import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PHASE 1: DATA CONFIGURATION
# ==========================================

# 1. Define Operational Parameters (Uncertainty Ranges)
# -----------------------------------------------------

# --- MODEL A: TRADITIONAL TRUCK ---
# Time estimates for manual operations (in Minutes)
TRUCK_OPS = {
    "loading": (15, 25),  # Loading at origin
    "unloading": (15, 25),  # Unloading at warehouse
    "last_mile": (10, 20),  # Final delivery to Tai Po Centre
}

# --- MODEL B: MTR + DRONE ---
# Time estimates for multi-modal steps (in Minutes)
MTR_OPS = {
    "first_mile": (5, 15),  # Walking/Van to origin station
    "wait_time": (0, 6),  # Waiting for the train (Headway)
    "transfer": (8, 12),  # Station to Drone Pad transfer
    "drone_load": (3, 5),  # Loading cargo into drone
    "flight": (3, 5),  # Flight to Tai Po Centre (1.8km)
}

# 2. Define Station Transit Data
# ------------------------------
# Origins: East Rail Line stations south of Tai Po Market.
# MTR Time: Fixed (Authoritative source).
# Truck Time: Range (Min, Max) based on Google Maps traffic data.

STATIONS_DB = {
    "Admiralty": {"mtr_time": 29, "truck_range": (35, 75)},
    "Exhibition Ctr": {"mtr_time": 27, "truck_range": (35, 70)},
    "Hung Hom": {"mtr_time": 22, "truck_range": (25, 55)},
    "Mong Kok East": {"mtr_time": 18, "truck_range": (20, 45)},
    "Kowloon Tong": {"mtr_time": 15, "truck_range": (20, 45)},
    "Tai Wai": {"mtr_time": 11, "truck_range": (15, 30)},
    "Sha Tin": {"mtr_time": 8, "truck_range": (12, 25)},
    "Fo Tan": {"mtr_time": 5, "truck_range": (10, 20)},
    "University": {"mtr_time": 3, "truck_range": (8, 15)},
}

# Verification Print
print("--- Phase 1: Data Loaded Successfully ---")
print(f"Total Stations Loaded: {len(STATIONS_DB)}")
print("Ready for Simulation Engine.")

# ==========================================
# PHASE 2: MONTE CARLO SIMULATION ENGINE
# ==========================================

# Settings
N_SIMULATIONS = 10000  # Run 10,000 scenarios for each station
np.random.seed(42)  # Fix random seed for reproducibility

# List to store simulation results
results_data = []

print(f"--- Starting Simulation Engine ({N_SIMULATIONS} runs per station) ---")

for station, data in STATIONS_DB.items():

    # -------------------------------------------------------
    # MODEL A: TRADITIONAL TRUCK (Using Uniform Distribution)
    # -------------------------------------------------------
    # Logic: We treat every step as a range with equal probability.

    # 1. Operational Steps (Loading, Unloading, Last Mile)
    t_load = np.random.uniform(
        TRUCK_OPS["loading"][0], TRUCK_OPS["loading"][1], N_SIMULATIONS
    )
    t_unload = np.random.uniform(
        TRUCK_OPS["unloading"][0], TRUCK_OPS["unloading"][1], N_SIMULATIONS
    )
    t_last = np.random.uniform(
        TRUCK_OPS["last_mile"][0], TRUCK_OPS["last_mile"][1], N_SIMULATIONS
    )

    # 2. Road Transit (The biggest uncertainty)
    # Uses the specific range for this station (e.g., 35-75 min for Admiralty)
    t_road = np.random.uniform(
        data["truck_range"][0], data["truck_range"][1], N_SIMULATIONS
    )

    # 3. Total Time Calculation (Vectorized Sum)
    total_time_truck = t_load + t_road + t_unload + t_last

    # -------------------------------------------------------
    # MODEL B: MTR + DRONE (Using Mixed Distributions)
    # -------------------------------------------------------
    # Logic: Operational steps are ranges (Uniform), but Rail Transit is fixed (Normal).

    # 1. Operational Steps (First Mile, Transfer, Drone Ops)
    t_first = np.random.uniform(
        MTR_OPS["first_mile"][0], MTR_OPS["first_mile"][1], N_SIMULATIONS
    )
    t_wait = np.random.uniform(
        MTR_OPS["wait_time"][0], MTR_OPS["wait_time"][1], N_SIMULATIONS
    )
    t_transfer = np.random.uniform(
        MTR_OPS["transfer"][0], MTR_OPS["transfer"][1], N_SIMULATIONS
    )
    t_drone_load = np.random.uniform(
        MTR_OPS["drone_load"][0], MTR_OPS["drone_load"][1], N_SIMULATIONS
    )
    t_flight = np.random.uniform(
        MTR_OPS["flight"][0], MTR_OPS["flight"][1], N_SIMULATIONS
    )

    # 2. MTR Rail Transit (The stable backbone)
    # Uses Normal Distribution centered at the official time with small variance (0.5 min)
    t_rail = np.random.normal(loc=data["mtr_time"], scale=0.5, size=N_SIMULATIONS)

    # 3. Total Time Calculation
    total_time_mtr = t_first + t_wait + t_rail + t_transfer + t_drone_load + t_flight

    # -------------------------------------------------------
    # DATA RECORDING
    # -------------------------------------------------------
    # We define "Worst Case" as the 95th percentile (P95) - important for logistics

    # Record Truck Stats
    results_data.append(
        {
            "Station": station,
            "Mode": "Traditional Truck",
            "Average_Time": np.mean(total_time_truck),
            "P95_Time": np.percentile(
                total_time_truck, 95
            ),  # 95% chance to arrive within this time
            "Std_Dev": np.std(total_time_truck),  # Stability metric
        }
    )

    # Record MTR Stats
    results_data.append(
        {
            "Station": station,
            "Mode": "MTR + Drone",
            "Average_Time": np.mean(total_time_mtr),
            "P95_Time": np.percentile(total_time_mtr, 95),
            "Std_Dev": np.std(total_time_mtr),
        }
    )

# Convert list to DataFrame for analysis
df_results = pd.DataFrame(results_data)

print("--- Simulation Complete ---")
print("Sample of calculated data (First 5 rows):")
print(df_results.head())


comparison_table = df_results.pivot(
    index="Station", columns="Mode", values="Average_Time"
)
comparison_table["Time Saved (min)"] = (
    comparison_table["Traditional Truck"] - comparison_table["MTR + Drone"]
)
comparison_table["Improvement (%)"] = (
    comparison_table["Time Saved (min)"] / comparison_table["Traditional Truck"]
) * 100
print("\n--- (Side-by-Side Comparison) ---")
print(
    comparison_table[
        ["Traditional Truck", "MTR + Drone", "Time Saved (min)", "Improvement (%)"]
    ].round(1)
)

# ==========================================
# PHASE 3: VISUALIZATION (CLEAN VERSION)
# ==========================================


def plot_simulation_results(df):
    # 1. Setup the plot style and size
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    # 2. Create a Grouped Bar Chart
    chart = sns.barplot(
        data=df,
        x="Station",
        y="Average_Time",
        hue="Mode",
        palette=["#d62728", "#1f77b4"],  # Red for Truck, Blue for MTR
        alpha=0.9,
        edgecolor="black",
    )

    # 3. Add Numerical Labels on Top of Bars
    for container in chart.containers:
        chart.bar_label(
            container, fmt="%.0f", padding=3, fontsize=11, fontweight="bold"
        )

    # 4. Chart Formatting
    plt.title(
        "Simulation Result: Traditional Truck vs. MTR + Drone\n(Impact of Origin Distance on Delivery Time)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    plt.ylabel("Average Total Time (Minutes)", fontsize=13, fontweight="bold")
    plt.xlabel("Origin Station (East Rail Line)", fontsize=13, fontweight="bold")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0, 140)
    plt.legend(title="Transport Mode", loc="upper right", frameon=True)

    # 5. Show the Plot
    plt.tight_layout()
    plt.show()


# Execute the visualization
print("--- Generating Final Comparison Chart (Clean)... ---")
plot_simulation_results(df_results)
