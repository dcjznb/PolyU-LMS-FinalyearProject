import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define Volume Range (From pilot 900 to massive 12000)
volumes = np.arange(900, 12500, 100)

# 2. Dynamic Model A Cost Function (Traditional Trucking)
truck_capacity_monthly = 2600
truck_fixed_opex = 26054 + 6000  # Driver + Parking
truck_variable_opex = 6220  # Fuel (simplified as per-truck base)
truck_cost_per_unit = truck_fixed_opex + truck_variable_opex


def calc_truck_cpd(v):
    num_trucks = np.ceil(v / truck_capacity_monthly)
    total_cost = num_trucks * truck_cost_per_unit
    return total_cost / v


truck_cpd_curve = [calc_truck_cpd(v) for v in volumes]

# 3. Dynamic Model B Cost Function (MTR + Drone)
operator_salary = 20800
mtr_fee = 3000
maint_per_drone = (2190 * 7.8) / 12
total_maint_drone = 4 * maint_per_drone

drone_fixed_cost = operator_salary + mtr_fee + total_maint_drone
# Variable: ~1.59 HKD battery + ~0.05 HKD electricity per parcel
drone_variable_cpd = 1.64

drone_cpd_curve = (drone_fixed_cost / volumes) + drone_variable_cpd

# 4. Plotting the Curves
sns.set_theme(style="whitegrid")
plt.figure(figsize=(11, 6.5))

plt.plot(
    volumes,
    truck_cpd_curve,
    label="Model A: Traditional Trucking (Step-Cost)",
    color="#d62728",
    linewidth=2.5,
)
plt.plot(
    volumes,
    drone_cpd_curve,
    label="Model B: MTR + Drone (Economy of Scale)",
    color="#1f77b4",
    linewidth=2.5,
)

# 5. Annotating the "Day-1 Dominance" at 900 parcels
pilot_vol = 900
truck_pilot_cpd = calc_truck_cpd(pilot_vol)
drone_pilot_cpd = (drone_fixed_cost / pilot_vol) + drone_variable_cpd

plt.scatter(
    [pilot_vol, pilot_vol],
    [truck_pilot_cpd, drone_pilot_cpd],
    color="purple",
    s=80,
    zorder=5,
)
plt.plot(
    [pilot_vol, pilot_vol],
    [drone_pilot_cpd, truck_pilot_cpd],
    color="purple",
    linestyle=":",
    linewidth=2,
)

plt.annotate(
    f"Pilot Stage (900/mo)\nTruck: {truck_pilot_cpd:.1f} HKD\nDrone: {drone_pilot_cpd:.1f} HKD",
    xy=(pilot_vol, drone_pilot_cpd),
    xytext=(pilot_vol + 500, drone_pilot_cpd - 5),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=7),
    fontsize=11,
    fontweight="bold",
    color="purple",
)

# 6. Formatting
plt.title(
    "Unit Economics (CPD) & Strategic Scalability\nTraditional Truck vs. MTR+Drone Logistics",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Monthly Delivery Volume (Parcels)", fontsize=12, fontweight="bold")
plt.ylabel("Cost Per Delivery (HKD / Parcel)", fontsize=12, fontweight="bold")
plt.xlim(900, 12000)
plt.ylim(0, 50)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()
