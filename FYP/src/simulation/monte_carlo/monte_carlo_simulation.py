# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation: Multi-Drone Delivery System
Parametric Analysis for Academic Publication
"""

import pandas as pd
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# =============================================================================
# CONFIGURATION & PARAMETERS
# =============================================================================

# We now test an array of fleet sizes to conduct sensitivity analysis
FLEET_SIZES = [1, 2, 3, 4, 5, 6, 8]  
NUM_SIMULATIONS = 100
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

stations = ["University", "Fo Tan", "Sha Tin", "Tai Wai", "Kowloon Tong", 
            "Mong Kok East", "Hung Hom", "Exhibition Centre", "Admiralty"]
density_scores = [3, 8, 13, 14, 18, 22, 18, 12, 12]
special_points = [10, 11, 14]

DRONE_MAX_WEIGHT = 30
DRONE_SPEED_KMH = 54
DRONE_MAX_RANGE_KM = 16
BATTERY_SWAP_TIME = 5
LOADING_TIME = 2
DROP_TIME = 1
SORTING_MU = 2.798
SORTING_SIGMA = 0.33

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def calc_distance(points, hub_lat, hub_lon):
    if not points: return 0
    cur_lat, cur_lon, total = hub_lat, hub_lon, 0
    points_copy = points.copy()
    while points_copy:
        min_dist, nearest_idx = float('inf'), 0
        for j, (lat, lon, _, _) in enumerate(points_copy):
            d = haversine(cur_lat, cur_lon, lat, lon)
            if d < min_dist:
                min_dist, nearest_idx = d, j
        nearest = points_copy.pop(nearest_idx)
        total += min_dist
        cur_lat, cur_lon = nearest[0], nearest[1]
    return total + haversine(cur_lat, cur_lon, hub_lat, hub_lon)

def generate_parcels():
    total = round(140 + random.uniform(-15, 15))
    special_ratio = random.uniform(0.30, 0.40)
    n_special = int(total * special_ratio)
    
    probs = np.array(density_scores) / sum(density_scores)
    items_per_station = np.random.multinomial(total, probs)
    
    assignments = [special_points[i % 3] for i in range(n_special)]
    regular_points = [p for p in range(1, 16) if p not in special_points]
    assignments += list(np.random.choice(regular_points, size=total - n_special, replace=True))
    np.random.shuffle(assignments)
    
    data = []
    for i, station in enumerate(stations):
        count = max(1, items_per_station[i])
        for _ in range(count):
            weight = round(random.uniform(0.5, 3.0), 2)
            dest = assignments[len(data) % len(assignments)]
            data.append((station, weight, dest))
    return pd.DataFrame(data, columns=['Station', 'Weight (kg)', 'Destination'])

def run_simulation(parcels, coords, hub, num_drones):
    grouped = parcels.groupby('Destination')['Weight (kg)'].sum().reset_index()
    batches, current_batch, current_weight = [], [], 0
    
    for _, row in grouped.iterrows():
        pid, weight = int(row['Destination']), row['Weight (kg)']
        if pid not in coords: continue
        lat, lon = coords[pid]['latitude'], coords[pid]['longitude']
        test_batch = current_batch + [(lat, lon, pid, weight)]
        test_dist = calc_distance(test_batch.copy(), hub[0], hub[1])
        
        if current_weight + weight <= DRONE_MAX_WEIGHT and test_dist <= DRONE_MAX_RANGE_KM:
            current_batch.append((lat, lon, pid, weight))
            current_weight += weight
        else:
            if current_batch: batches.append(current_batch)
            current_batch, current_weight = [(lat, lon, pid, weight)], weight
    if current_batch: batches.append(current_batch)
    
    sorting_time = np.random.lognormal(mean=SORTING_MU, sigma=SORTING_SIGMA)
    
    drones = [{'id': i+1, 'available': sorting_time, 'completion': 0, 'swaps': 0, 
               'flight_time': 0, 'load_time': 0, 'drop_time': 0} for i in range(num_drones)]
    
    batch_timeline = []
    parcel_lead_times = [] # To track customer-centric delivery times
    total_flight_distance = 0
    
    for batch_idx, batch in enumerate(batches, 1):
        drone = min(drones, key=lambda d: d['available'])
        is_first_flight = (drone['available'] == sorting_time)
        swap_duration = 0 if is_first_flight else BATTERY_SWAP_TIME
        
        load_start = drone['available'] + swap_duration
        load_end = load_start + LOADING_TIME
        
        dist = calc_distance(batch.copy(), hub[0], hub[1])
        total_flight_distance += dist
        outbound_dist = dist * 0.85
        return_dist = dist * 0.15
        
        flight_out = outbound_dist / DRONE_SPEED_KMH * 60
        flight_end = load_end + flight_out
        
        drop_time = len(batch) * DROP_TIME
        drop_end = flight_end + drop_time
        
        flight_back = return_dist / DRONE_SPEED_KMH * 60
        return_end = drop_end + flight_back
        
        # Record lead time for each parcel in this batch
        for _ in range(len(batch)):
            parcel_lead_times.append(drop_end)
        
        batch_timeline.append({
            'batch_id': batch_idx,
            'drone_id': drone['id'],
            'swap_start': drone['available'] if not is_first_flight else None,
            'load_start': load_start,
            'load_end': load_end,
            'flight_out_end': flight_end,
            'drop_end': drop_end,
            'return_end': return_end
        })
        
        if not is_first_flight: drone['swaps'] += 1
        drone['load_time'] += LOADING_TIME
        drone['flight_time'] += (flight_out + flight_back)
        drone['drop_time'] += drop_time
        drone['available'] = return_end
        drone['completion'] = return_end 
    
    makespan = max(d['completion'] for d in drones)
    avg_parcel_lead = sum(parcel_lead_times) / len(parcel_lead_times) if parcel_lead_times else 0
    
    # Calculate Utilization Metrics
    active_time_sum = sum((d['flight_time'] + d['load_time'] + d['drop_time'] + (d['swaps']*BATTERY_SWAP_TIME)) for d in drones)
    total_available_time = num_drones * (makespan - sorting_time)
    utilization_rate = (active_time_sum / total_available_time) * 100 if total_available_time > 0 else 0
    
    return {
        'total_time': float(makespan),
        'avg_parcel_lead': float(avg_parcel_lead),
        'utilization_rate': float(utilization_rate),
        'total_flight_dist': float(total_flight_distance),
        'sorting': float(sorting_time),
        'n_batches': int(len(batches)),
        'timeline': batch_timeline
    }

# =============================================================================
# MAIN EXECUTION & PARAMETER EXTRACTION
# =============================================================================

# REPLACE THIS PATH WITH YOUR ACTUAL FILE PATH
EXCEL_PATH = r"C:\Users\Morgan MA\Desktop\223\Map coordinate.xlsx"

coords_df = pd.read_excel(EXCEL_PATH)
coords_df[['latitude', 'longitude']] = coords_df['coordinate'].str.split(',', expand=True).astype(float)

hub_row = coords_df[coords_df['Function'] == 'Drone Station （Start）']
hub = (hub_row['latitude'].values[0], hub_row['longitude'].values[0])

coords = {}
for i in range(1, 16):
    row = coords_df[coords_df['Function'] == f'Destination {i}']
    if len(row) > 0:
        coords[i] = {'latitude': float(row['latitude'].values[0]), 'longitude': float(row['longitude'].values[0])}

print("="*85)
print(f"{'ACADEMIC PARAMETER OUTPUT: SENSITIVITY ANALYSIS OVER FLEET SIZES':^85}")
print("="*85)
print(f"{'Drones':<8} | {'Mean Makespan':<15} | {'P95 Makespan':<14} | {'Avg Parcel Lead':<16} | {'Utilization':<12} | {'Total Dist (km)'}")
print("-" * 85)

all_results = {}

for num_drones in FLEET_SIZES:
    # Run simulation for this fleet size
    res = [run_simulation(generate_parcels(), coords, hub, num_drones) for _ in range(NUM_SIMULATIONS)]
    all_results[num_drones] = res
    
    # Extract valuable parameters for the console table
    makespans = [r['total_time'] for r in res]
    mean_makespan = np.mean(makespans)
    p95_makespan = np.percentile(makespans, 95)
    mean_lead = np.mean([r['avg_parcel_lead'] for r in res])
    mean_util = np.mean([r['utilization_rate'] for r in res])
    mean_dist = np.mean([r['total_flight_dist'] for r in res])
    
    print(f"{num_drones:<8} | {mean_makespan:>9.1f} min    | {p95_makespan:>9.1f} min  | {mean_lead:>10.1f} min    | {mean_util:>9.1f} %  | {mean_dist:>9.1f} km")

print("="*85)

# =============================================================================
# MULTI-DIMENSIONAL CHART GENERATION
# =============================================================================
print("\nGenerating comprehensive academic charts...")

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
charts_dir = os.path.join(script_dir, 'charts')
os.makedirs(charts_dir, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

# -----------------------------------------------------------------------------
# CHART 1: Diminishing Marginal Returns (Makespan vs. Fleet Size)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
means = [np.mean([r['total_time'] for r in all_results[d]]) for d in FLEET_SIZES]
p95s = [np.percentile([r['total_time'] for r in all_results[d]], 95) for d in FLEET_SIZES]
p05s = [np.percentile([r['total_time'] for r in all_results[d]], 5) for d in FLEET_SIZES]

ax.plot(FLEET_SIZES, means, marker='o', color='darkblue', linewidth=2, label='Mean Makespan')
ax.fill_between(FLEET_SIZES, p05s, p95s, color='royalblue', alpha=0.2, label='90% Confidence Interval (P05-P95)')
ax.plot(FLEET_SIZES, p95s, linestyle='--', color='darkred', alpha=0.7, label='P95 Makespan (Worst-Case)')

ax.set_xlabel('Drone Fleet Size', fontsize=12)
ax.set_ylabel('Total Completion Time / Makespan (minutes)', fontsize=12)
ax.set_title('Diminishing Marginal Returns in Fleet Expansion', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
plt.savefig(os.path.join(charts_dir, 'Chart_1_Makespan_vs_FleetSize.png'), dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# CHART 2: Boxplot of Makespan Variance
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
plot_data = [ [r['total_time'] for r in all_results[d]] for d in FLEET_SIZES ]
sns.boxplot(data=plot_data, palette="Blues_d", ax=ax)
ax.set_xticklabels(FLEET_SIZES)
ax.set_xlabel('Drone Fleet Size', fontsize=12)
ax.set_ylabel('Total Completion Time (minutes)', fontsize=12)
ax.set_title('Distribution & Volatility of Delivery Completion Times', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(charts_dir, 'Chart_2_Makespan_Boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# CHART 3: Drone Fleet Utilization vs Customer Parcel Lead Time
# -----------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 6))

utils = [np.mean([r['utilization_rate'] for r in all_results[d]]) for d in FLEET_SIZES]
leads = [np.mean([r['avg_parcel_lead'] for r in all_results[d]]) for d in FLEET_SIZES]

ax1.bar(FLEET_SIZES, utils, color='teal', alpha=0.7, width=0.6, label='Fleet Utilization (%)')
ax1.set_xlabel('Drone Fleet Size', fontsize=12)
ax1.set_ylabel('Fleet Utilization Rate (%)', color='teal', fontsize=12)
ax1.tick_params(axis='y', labelcolor='teal')
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()
ax2.plot(FLEET_SIZES, leads, color='darkorange', marker='s', linewidth=2.5, label='Avg Parcel Lead Time (min)')
ax2.set_ylabel('Average Customer Parcel Wait Time (minutes)', color='darkorange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(0, max(leads)*1.2)

plt.title('Trade-off: Capital Efficiency (Utilization) vs Service Level (Lead Time)', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(charts_dir, 'Chart_3_Utilization_vs_LeadTime.png'), dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# CHART 4 & 5: Gantt Chart and System Utilization Profile (Using Fleet Size = 4)
# -----------------------------------------------------------------------------
FOCUS_DRONES = 4
if FOCUS_DRONES in FLEET_SIZES:
    focus_results = all_results[FOCUS_DRONES]
    typical_idx = np.argmin([abs(r['total_time'] - np.mean([x['total_time'] for x in focus_results])) for r in focus_results])
    typical = focus_results[typical_idx]
    
    COLORS = {'sorting': '#95a5a6', 'loading': '#3498db', 'flight': '#e67e22', 
              'drop': '#e74c3c', 'battery': '#9b59b6', 'return': '#f1c40f'}
    
    # Gantt Chart
    fig, ax = plt.subplots(figsize=(12, max(5, FOCUS_DRONES * 1.2)))
    ax.barh("Central Hub", typical['sorting'], left=0, color=COLORS['sorting'], edgecolor='black', height=0.5)
    ax.text(typical['sorting']/2, "Central Hub", f'Sorting\n{typical["sorting"]:.1f}m', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    for batch in typical['timeline']:
        y_pos = f"Drone {batch['drone_id']}"
        if batch['swap_start'] is not None:
            ax.barh(y_pos, BATTERY_SWAP_TIME, left=batch['swap_start'], color=COLORS['battery'], edgecolor='white', height=0.6)
        ax.barh(y_pos, batch['load_end'] - batch['load_start'], left=batch['load_start'], color=COLORS['loading'], edgecolor='white', height=0.6)
        ax.barh(y_pos, batch['flight_out_end'] - batch['load_end'], left=batch['load_end'], color=COLORS['flight'], edgecolor='white', height=0.6)
        ax.barh(y_pos, batch['drop_end'] - batch['flight_out_end'], left=batch['flight_out_end'], color=COLORS['drop'], edgecolor='white', height=0.6)
        ax.barh(y_pos, batch['return_end'] - batch['drop_end'], left=batch['drop_end'], color=COLORS['return'], edgecolor='white', height=0.6)
        ax.text(batch['load_start'] + (batch['return_end'] - batch['load_start'])/2, y_pos, f"B{batch['batch_id']}", ha='center', va='center', fontsize=8, color='black', fontweight='bold')

    ax.axvline(x=typical['total_time'], color='red', linestyle='--', linewidth=2)
    ax.text(typical['total_time'] + 0.5, "Central Hub", f'Makespan\n{typical["total_time"]:.1f} m', ha='left', va='center', fontsize=11, color='red', fontweight='bold')

    legend_elements = [Patch(facecolor=COLORS[k], label=k.capitalize()) for k in ['sorting', 'battery', 'loading', 'flight', 'drop', 'return']]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=10)
    ax.set_xlabel('Elapsed Time (minutes)', fontsize=12)
    ax.set_title(f'Drone Scheduling Gantt Chart (Critical Path) - {FOCUS_DRONES} Drones', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, f'Chart_4_Gantt_{FOCUS_DRONES}_Drones.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"\n[Success] Analysis complete. 4 unique charts generated in the '{charts_dir}' directory.")