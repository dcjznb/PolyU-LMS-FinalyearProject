import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# PART 1: STATIC FINANCIAL AUDIT (Target Volume = 10,500 parcels/month)
# ==========================================

print("=" * 60)
print("FINANCIAL AUDIT: TCO & UNIT ECONOMICS ANALYSIS")
print("Target Monthly Volume: 10,500 parcels (420 parcels/day)")
print("=" * 60)

target_volume = 10500

# --- MODEL A: TRADITIONAL TRUCK PARAMETERS ---
truck_capacity_monthly = 2625  # ~105 parcels/day * 25 days
trucks_needed = int(np.ceil(target_volume / truck_capacity_monthly))  # 4 trucks

driver_salary = 26054
fuel_cost_per_truck = (
    (8.5 / 100) * 100 * 25 * 29.27
)  # 8.5L/100km, 100km/day, 25 days, 29.27 HKD/L
parking_toll_per_truck = 6000

# Model A Calculations
total_labor_truck = trucks_needed * driver_salary
total_fuel_truck = trucks_needed * fuel_cost_per_truck
total_parking_truck = trucks_needed * parking_toll_per_truck

total_opex_truck = total_labor_truck + total_fuel_truck + total_parking_truck
cpd_truck = total_opex_truck / target_volume

# --- MODEL B: MTR + DRONE HYBRID PARAMETERS ---
drones_needed = 4
flights_needed = 1050  # Assuming 10 parcels per batch

operator_salary = 20800
battery_replacement_cost = 3183
battery_lifecycle = 200
cost_per_flight_battery = battery_replacement_cost / battery_lifecycle

maint_per_drone = (2190 * 7.8) / 12  # 2190 USD/year to HKD/month
electricity_total = 580
mtr_fee = 3000

# Model B Calculations
total_battery_drone = flights_needed * cost_per_flight_battery
total_maint_drone = drones_needed * maint_per_drone

total_opex_drone = (
    operator_salary
    + total_battery_drone
    + electricity_total
    + mtr_fee
    + total_maint_drone
)
cpd_drone = total_opex_drone / target_volume

# --- DISPLAY STATIC RESULTS ---
data = {
    "Cost Category (HKD/month)": [
        "Labor",
        "Energy (Fuel / Elec)",
        "Infrastructure / Parking",
        "Maintenance / Battery",
        "TOTAL OPEX",
        "CPD (Cost Per Delivery)",
    ],
    "Model A (4 Trucks)": [
        f"${total_labor_truck:,.0f}",
        f"${total_fuel_truck:,.0f}",
        f"${total_parking_truck:,.0f}",
        "N/A (in CapEx)",
        f"${total_opex_truck:,.0f}",
        f"${cpd_truck:.2f} / parcel",
    ],
    "Model B (MTR + 4 Drones)": [
        f"${operator_salary:,.0f}",
        f"${electricity_total:,.0f}",
        f"${mtr_fee:,.0f}",
        f"${(total_battery_drone + total_maint_drone):,.0f}",
        f"${total_opex_drone:,.0f}",
        f"${cpd_drone:.2f} / parcel",
    ],
}

df_financials = pd.DataFrame(data)
print("\n[COST BREAKDOWN TABLE]")
print(df_financials.to_string(index=False))
print("-" * 60)
print(
    f"CONCLUSION: At {target_volume} parcels, Model B saves ${(cpd_truck - cpd_drone):.2f} per parcel."
)
print("=" * 60)


# ==========================================
# PART 2: DYNAMIC BREAK-EVEN ANALYSIS (1,000 to 12,000 parcels)
# ==========================================

# 1. Define Volume Range
volumes = np.arange(1000, 12500, 100)

# 2. Dynamic Model A Cost Function (Step-Cost)
truck_cost_per_unit = driver_salary + fuel_cost_per_truck + parking_toll_per_truck


def calc_truck_cpd(v):
    num_trucks = np.ceil(
        v / truck_capacity_monthly
    )  # Scales up when capacity is breached
    total_cost = num_trucks * truck_cost_per_unit
    return total_cost / v


truck_cpd_curve = [calc_truck_cpd(v) for v in volumes]

# 3. Dynamic Model B Cost Function (Economy of Scale)
# Fixed Costs: Salary + MTR Fee + Maintenance
drone_fixed_cost = operator_salary + mtr_fee + total_maint_drone
# Variable Costs: Battery + Electricity per parcel
drone_variable_cpd = (total_battery_drone + electricity_total) / target_volume

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

# 5. Locate and Annotate Break-even Point
idx = np.argmin(np.abs(np.array(truck_cpd_curve) - np.array(drone_cpd_curve)))
be_volume = volumes[idx]
be_price = drone_cpd_curve[idx]

plt.scatter(be_volume, be_price, color="purple", s=100, zorder=5)
plt.axvline(x=be_volume, color="purple", linestyle="--", alpha=0.7)
plt.axhline(y=be_price, color="purple", linestyle="--", alpha=0.7)

plt.annotate(
    f"Break-even Point\n~{int(be_volume)} parcels/mo\n@ {be_price:.1f} HKD/parcel",
    xy=(be_volume, be_price),
    xytext=(be_volume + 600, be_price + 8),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=7),
    fontsize=11,
    fontweight="bold",
    color="purple",
)

# 6. Formatting
plt.title(
    "Unit Economics (CPD) & Break-even Analysis\nTraditional Truck vs. MTR+Drone Logistics",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Monthly Delivery Volume (Parcels)", fontsize=12, fontweight="bold")
plt.ylabel("Cost Per Delivery (HKD / Parcel)", fontsize=12, fontweight="bold")
plt.xlim(1000, 12000)
plt.ylim(0, 50)
plt.legend(fontsize=11)
plt.tight_layout()

# Show plot
plt.show()
