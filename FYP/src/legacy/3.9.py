# -*- coding: utf-8 -*-
"""
Monte Carlo Simulation: Multi-Drone Delivery System
Author: Morgan MA
Date: March 2026
"""

import pandas as pd
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_DRONES = 4  # Dynamic drone fleet size
NUM_SIMULATIONS = 100
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# =============================================================================
# PARAMETERS
# =============================================================================

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
    """Calculate great-circle distance between two points in km"""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def calc_distance(points, hub_lat, hub_lon):
    """Calculate route distance using nearest-neighbor algorithm"""
    if not points:
        return 0
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
    """Generate random parcel data with specified distribution"""
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
    """Run single simulation and return timing metrics"""
    grouped = parcels.groupby('Destination')['Weight (kg)'].sum().reset_index()
    batches, current_batch, current_weight = [], [], 0
    
    for _, row in grouped.iterrows():
        pid, weight = int(row['Destination']), row['Weight (kg)']
        if pid not in coords:
            continue
        lat, lon = coords[pid]['latitude'], coords[pid]['longitude']
        test_batch = current_batch + [(lat, lon, pid, weight)]
        test_dist = calc_distance(test_batch.copy(), hub[0], hub[1])
        
        if current_weight + weight <= DRONE_MAX_WEIGHT and test_dist <= DRONE_MAX_RANGE_KM:
            current_batch.append((lat, lon, pid, weight))
            current_weight += weight
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch, current_weight = [(lat, lon, pid, weight)], weight
    if current_batch:
        batches.append(current_batch)
    
    sorting_time = np.random.lognormal(mean=SORTING_MU, sigma=SORTING_SIGMA)
    drones = [{'available': sorting_time, 'completion': 0, 'swaps': 0} for _ in range(num_drones)]
    
    total_flight_dist = 0
    batch_timeline = []
    
    for batch_idx, batch in enumerate(batches, 1):
        drone = min(drones, key=lambda d: d['available'])
        load_start = drone['available']
        load_end = load_start + LOADING_TIME
        
        dist = calc_distance(batch.copy(), hub[0], hub[1])
        total_flight_dist += dist
        return_dist = dist * 0.15
        outbound_dist = dist - return_dist
        
        flight_out = outbound_dist / DRONE_SPEED_KMH * 60
        flight_end = load_end + flight_out
        
        drop_time = len(batch) * DROP_TIME
        drop_end = flight_end + drop_time
        
        flight_back = return_dist / DRONE_SPEED_KMH * 60
        return_end = drop_end + flight_back
        
        batch_timeline.append({
            'batch_id': batch_idx,
            'drone_id': drone['id'] if 'id' in drone else drones.index(drone) + 1,
            'load_start': load_start,
            'load_end': load_end,
            'flight_out_end': flight_end,
            'drop_end': drop_end,
            'return_end': return_end,
            'n_points': len(batch)
        })
        
        if len(batches) > num_drones:
            drone['swaps'] += 1
            drone['available'] = return_end + BATTERY_SWAP_TIME
        else:
            drone['available'] = return_end
        drone['completion'] = drop_end
    
    total_time = max(d['completion'] for d in drones)
    total_battery = sum(d['swaps'] for d in drones) * BATTERY_SWAP_TIME
    
    return {
        'total_time': float(total_time),
        'sorting': float(sorting_time),
        'loading': float(len(batches) * LOADING_TIME),
        'flight': float(total_flight_dist / DRONE_SPEED_KMH * 60),
        'drop': float(sum(len(b) for b in batches) * DROP_TIME),
        'battery': float(total_battery),
        'n_batches': int(len(batches)),
        'n_parcels': int(len(parcels)),
        'timeline': batch_timeline
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("=" * 70)
print("MONTE CARLO SIMULATION | Drone Fleet Size: {} | Runs: {}".format(
    NUM_DRONES, NUM_SIMULATIONS))
print("=" * 70)

# Load coordinates
coords_df = pd.read_excel(r"C:\Users\Morgan MA\Desktop\223\Map coordinate.xlsx")
coords_df[['latitude', 'longitude']] = coords_df['coordinate'].str.split(',', expand=True).astype(float)

hub_row = coords_df[coords_df['Function'] == 'Drone Station （Start）']
hub = (hub_row['latitude'].values[0], hub_row['longitude'].values[0])

coords = {}
for i in range(1, 16):
    row = coords_df[coords_df['Function'] == 'Destination {}'.format(i)]
    if len(row) > 0:
        coords[i] = {'latitude': float(row['latitude'].values[0]), 
                     'longitude': float(row['longitude'].values[0])}

print("Coordinates loaded: {} delivery points".format(len(coords)))

# Run simulations
print("Running {} simulations...".format(NUM_SIMULATIONS))
results = [run_simulation(generate_parcels(), coords, hub, NUM_DRONES) 
           for _ in range(NUM_SIMULATIONS)]
print("Completed.\n")

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def calc_stats(data):
    """Calculate descriptive statistics"""
    return {
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'p95': float(np.percentile(data, 95)),
        'p50': float(np.percentile(data, 50)),
        'p90': float(np.percentile(data, 90))
    }

metrics = ['total_time', 'sorting', 'loading', 'flight', 'drop', 'battery', 'n_batches', 'n_parcels']
stats_dict = {m: calc_stats([r[m] for r in results]) for m in metrics}
t = stats_dict['total_time']

# Step 1: Print key timing statistics
print("=" * 70)
print("KEY TIMING STATISTICS (Drone Fleet Size: {})".format(NUM_DRONES))
print("=" * 70)

print("\n[1] Total Completion Time (minutes)")
print("-" * 70)
print("  Mean:                     {:>8.1f}".format(t['mean']))
print("  Median:                   {:>8.1f}".format(t['median']))
print("  Std Dev:                  {:>8.1f}".format(t['std']))
print("  Min:                      {:>8.1f}".format(t['min']))
print("  Max:                      {:>8.1f}".format(t['max']))
print("  P95:                      {:>8.1f}".format(t['p95']))

cv = t['std'] / t['mean'] * 100
print("\n[2] Derived Metrics")
print("-" * 70)
print("  P95 Completion Time:      {:>8.1f} minutes".format(t['p95']))
print("  Coefficient of Variation: {:>8.1f}%".format(cv))

flight_ratio = stats_dict['flight']['mean'] / t['mean'] * 100
battery_ratio = stats_dict['battery']['mean'] / t['mean'] * 100
avg_batch_interval = t['mean'] / stats_dict['n_batches']['mean']

print("  Flight Time Ratio:        {:>8.1f}%".format(flight_ratio))
print("  Battery Swap Time (mean): {:>8.1f} minutes ({:.1f}%)".format(
    stats_dict['battery']['mean'], battery_ratio))
print("  Avg Batch Completion Interval: {:>5.1f} minutes/batch".format(avg_batch_interval))

# =============================================================================
# CHART GENERATION
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING CHARTS")
print("=" * 70)

# Setup output directory
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
charts_dir = os.path.join(script_dir, 'charts')
os.makedirs(charts_dir, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Chart A: Completion Time Distribution (Histogram + KDE)
print("Generating Chart A: Completion Time Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))
times = [r['total_time'] for r in results]
ax.hist(times, bins=25, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Histogram')
sns.kdeplot(times, ax=ax, color='darkred', linewidth=2, label='KDE')

ax.axvline(t['mean'], color='green', linestyle='--', linewidth=1.5, label='Mean: {:.1f} min'.format(t['mean']))
ax.axvline(t['median'], color='blue', linestyle='--', linewidth=1.5, label='Median: {:.1f} min'.format(t['median']))
ax.axvline(t['p95'], color='red', linestyle='--', linewidth=1.5, label='P95: {:.1f} min'.format(t['p95']))

ax.set_xlabel('Completion Time (minutes)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Completion Time Distribution\n(Drone Fleet Size: {}, {} Simulations)'.format(
    NUM_DRONES, NUM_SIMULATIONS), fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'A_completion_time_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: A_completion_time_distribution.png")

# Chart B: CDF of Completion Time
print("Generating Chart B: CDF of Completion Time...")
fig, ax = plt.subplots(figsize=(10, 6))
sorted_times = np.sort(times)
cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
ax.plot(sorted_times, cdf, linewidth=2, color='navy')

ax.axvline(t['p50'], color='green', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(t['p90'], color='orange', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(t['p95'], color='red', linestyle=':', linewidth=1, alpha=0.7)

ax.text(t['p50'], 0.52, 'P50\n{:.1f}'.format(t['p50']), ha='center', fontsize=9, color='green')
ax.text(t['p90'], 0.92, 'P90\n{:.1f}'.format(t['p90']), ha='center', fontsize=9, color='orange')
ax.text(t['p95'], 0.97, 'P95\n{:.1f}'.format(t['p95']), ha='center', fontsize=9, color='red')

ax.set_xlabel('Completion Time (minutes)', fontsize=11)
ax.set_ylabel('Cumulative Probability', fontsize=11)
ax.set_title('CDF of Completion Time\n(Drone Fleet Size: {})'.format(NUM_DRONES), 
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'B_completion_time_cdf.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: B_completion_time_cdf.png")

# Chart C: Time Breakdown Stacked Bar
print("Generating Chart C: Time Breakdown...")
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['sorting', 'loading', 'flight', 'drop', 'battery']
labels = ['Sorting', 'Loading', 'Flight', 'Drop-off', 'Battery Swap']
colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6']
values = [stats_dict[c]['mean'] for c in categories]

bottom = 0
for i, (val, label, color) in enumerate(zip(values, labels, colors)):
    ax.bar(0, val, bottom=bottom, color=color, label=label, edgecolor='white', linewidth=1)
    if val > 2:
        ax.text(0, bottom + val/2, '{:.1f}\n({:.1f}%)'.format(val, val/t['mean']*100), 
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    bottom += val

ax.set_xlim(-0.5, 0.5)
ax.set_xticks([])
ax.set_ylabel('Time (minutes)', fontsize=11)
ax.set_title('Time Breakdown - Mean Values\n(Drone Fleet Size: {})'.format(NUM_DRONES), 
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'C_time_breakdown_stacked.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: C_time_breakdown_stacked.png")

# Chart D: Gantt Chart for Typical Simulation
print("Generating Chart D: Gantt Chart...")

# Find simulation closest to mean total time
typical_idx = np.argmin([abs(r['total_time'] - t['mean']) for r in results])
typical = results[typical_idx]
timeline = typical['timeline']

fig, ax = plt.subplots(figsize=(14, 8))

# Color scheme
COLORS = {'sorting': '#2ecc71', 'loading': '#3498db', 'flight': '#e67e22', 
          'drop': '#e74c3c', 'battery': '#9b59b6', 'return': '#27ae60'}

# Plot sorting phase
ax.barh(0, typical['sorting'], left=0, color=COLORS['sorting'], edgecolor='white', label='Sorting')
ax.text(typical['sorting']/2, 0, 'Sorting\n{:.1f}min'.format(typical['sorting']), 
        ha='center', va='center', fontsize=9, color='white', fontweight='bold')

# Plot batches (show first 8 if more exist)
max_batches_display = min(8, len(timeline))
y_positions = np.linspace(-1.5, -1.5 - (max_batches_display-1)*0.9, max_batches_display)

for i, (batch, y_pos) in enumerate(zip(timeline[:max_batches_display], y_positions)):
    batch_id = batch['batch_id']
    
    # Loading
    ax.barh(y_pos, batch['load_end'] - batch['load_start'], 
            left=batch['load_start'], color=COLORS['loading'], edgecolor='white', height=0.7)
    
    # Flight out
    ax.barh(y_pos, batch['flight_out_end'] - batch['load_end'], 
            left=batch['load_end'], color=COLORS['flight'], edgecolor='white', height=0.7)
    
    # Drop-off
    ax.barh(y_pos, batch['drop_end'] - batch['flight_out_end'], 
            left=batch['flight_out_end'], color=COLORS['drop'], edgecolor='white', height=0.7)
    
    # Return flight
    ax.barh(y_pos, batch['return_end'] - batch['drop_end'], 
            left=batch['drop_end'], color=COLORS['return'], edgecolor='white', height=0.7, alpha=0.7)
    
    # Battery swap (if applicable)
    if batch_id < len(timeline):
        ax.barh(y_pos, BATTERY_SWAP_TIME, left=batch['return_end'], 
                color=COLORS['battery'], edgecolor='white', height=0.7)
    
    # Label
    ax.text((batch['load_start'] + batch['drop_end'])/2, y_pos, 
            'Batch {}'.format(batch_id), ha='center', va='center', 
            fontsize=9, color='white', fontweight='bold')

# Vertical lines for key milestones
ax.axvline(x=typical['sorting'], color='black', linestyle=':', linewidth=1, alpha=0.5)
ax.axvline(x=typical['total_time'], color='red', linestyle='-', linewidth=2, label='End-to-End Time')
ax.text(typical['total_time'], -max_batches_display*0.9 - 0.5, 
        'TOTAL: {:.1f} min'.format(typical['total_time']), 
        ha='center', va='top', fontsize=11, color='red', fontweight='bold')

# Labels and legend
ax.set_xlabel('Time (minutes)', fontsize=11)
ax.set_ylabel('Operation Phase', fontsize=11)
ax.set_title('Example Delivery Timeline Gantt Chart\n(Drone Fleet Size: {}, Typical Simulation)'.format(
    NUM_DRONES), fontsize=13, fontweight='bold')

ytick_labels = ['Hub Processing'] + ['Batch {}'.format(timeline[i]['batch_id']) for i in range(max_batches_display)]
ax.set_yticks([0] + list(y_positions))
ax.set_yticklabels(ytick_labels, fontsize=10)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS['sorting'], label='Sorting'),
    Patch(facecolor=COLORS['loading'], label='Loading'),
    Patch(facecolor=COLORS['flight'], label='Flight (Outbound)'),
    Patch(facecolor=COLORS['drop'], label='Drop-off'),
    Patch(facecolor=COLORS['return'], label='Return Flight'),
    Patch(facecolor=COLORS['battery'], label='Battery Swap')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_xlim(0, typical['total_time'] * 1.1)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, 'D_delivery_timeline_gantt.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: D_delivery_timeline_gantt.png")

print("\nAll charts saved to:", charts_dir)

# =============================================================================
# EXPORT RESULTS
# =============================================================================

output_data = {
    'configuration': {
        'num_drones': int(NUM_DRONES),
        'num_simulations': int(NUM_SIMULATIONS),
        'random_seed': int(RANDOM_SEED)
    },
    'statistics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in stats_dict.items()},
    'derived_metrics': {
        'cv_percent': float(cv),
        'flight_time_ratio_percent': float(flight_ratio),
        'battery_time_ratio_percent': float(battery_ratio),
        'avg_batch_interval_min': float(avg_batch_interval)
    }
}

with open(os.path.join(script_dir, 'monte_carlo_results.json'), 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("\n[EXPORT] Results saved to: monte_carlo_results.json")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)