import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PHASE 1: DATA CONSOLIDATION
# ==========================================

# Defining the raw data dictionary.
# This serves as the database for our comparative analysis.
raw_data = {
    # 1. Area Names
    # We selected 4 distinct scenarios to represent different urban morphologies in Hong Kong.
    "Area": [
        "Tai Po (Baseline)",  # Target: New Town with balanced features
        "Hung Hom (High Density)",  # Control A: Too dense, ground traffic is static
        "Sai Kung (Remote)",  # Control B: Too far, lacks infrastructure
        "Mid-Levels (Hilly)",  # Control C: Vertical city, high terrain value
    ],
    # 2. Safety Metric: Population Density
    # Unit: Person per sq. km
    # Source: 2021 Population Census (District & Constituency level)
    # Logic: Lower density implies lower ground risk and easier flight compliance.
    "Population_Density": [2137, 100779.9239, 11115, 18967],
    # 3. Efficiency Metrics: Distance Components
    # Unit: Kilometers (km)
    # Source: Google Maps (Driving vs Linear measurement)
    # Logic: A high ratio of (Driving / Linear) indicates high "tortuosity",
    # meaning drones can significantly shorten the path (e.g., in Hilly areas).
    "Driving_Distance": [2.5, 0.4, 11.0, 2.5],  # Traditional truck route
    "Linear_Distance": [1.8, 0.3, 8.5, 0.6],  # Drone flight path
    # 4. Necessity Metric: Congestion Index
    # Unit: Ratio (Peak Hour Time / Free Flow Time)
    # Source: Google Maps Real-time Traffic Data
    # Logic: Higher index (>1.5) means severe ground congestion, validating the need for aerial solutions.
    # Data points: Tai Po(1.27), Hung Hom(1.0), Sai Kung(1.33), Mid-Levels(1.818)
    "Congestion_Index": [1.27, 1.0, 1.33, 1.818],
}

# Verification: Print the raw data to ensure correct entry
print("--- Phase 1: Data Loaded Successfully ---")
# Create a temporary DataFrame to check the table structure
df_preview = pd.DataFrame(raw_data)
print(df_preview)

# ==========================================
# PHASE 2: MULTI-SCENARIO SCORING ENGINE
# ==========================================


def calculate_scenarios(data_dict):
    df = pd.DataFrame(data_dict)

    # -----------------------------------------------------
    # 1. Calculate Basic Component Scores (0-10)
    # -----------------------------------------------------

    # Demand (Based on Density)
    d = df["Population_Density"]
    demand = (d - d.min()) / (d.max() - d.min()) * 10

    # Necessity (Based on Congestion)
    c = df["Congestion_Index"]
    necessity = (c - c.min()) / (c.max() - c.min()) * 10

    # Efficiency (Based on Tortuosity & Range)
    tortuosity = df["Driving_Distance"] / (df["Linear_Distance"] + 0.01)
    range_penalty = np.where(df["Linear_Distance"] > 5.0, 0.2, 1.0)
    raw_eff = tortuosity * range_penalty
    efficiency = (raw_eff - raw_eff.min()) / (raw_eff.max() - raw_eff.min()) * 10

    # -----------------------------------------------------
    # 2. Define Scenarios (The "Menu" for Leaders)
    # -----------------------------------------------------
    # Format: [Efficiency_Weight, Demand_Weight, Necessity_Weight]

    scenarios = {
        "1. Cost Focus\n(Efficiency 60%)": [
            0.6,
            0.2,
            0.2,
        ],
        "2. Market Scale\n(Demand 60%)": [0.2, 0.6, 0.2],
        "3. Pain Point\n(Necessity 60%)": [0.2, 0.2, 0.6],
        "4. Balanced\n(Equal Weights)": [0.33, 0.33, 0.33],
    }

    # -----------------------------------------------------
    # 3. Calculate Scores for Each Scenario
    # -----------------------------------------------------
    results = df[["Area"]].copy()

    for name, weights in scenarios.items():
        w_eff, w_dem, w_nec = weights
        # Calculate weighted sum
        score = (efficiency * w_eff) + (demand * w_dem) + (necessity * w_nec)
        results[name] = score

    return results


# Execute Calculation
scenario_scores = calculate_scenarios(raw_data)

# Print Table for verification
print("\n--- Multi-Scenario Analysis Results ---")
print(scenario_scores.round(2))


# ==========================================
# PHASE 3: VISUALIZATION (DECISION HEATMAP)
# ==========================================


def plot_decision_heatmap(scores_df):
    # Prepare data for Heatmap
    # Set 'Area' as index so rows are areas
    heatmap_data = scores_df.set_index("Area")

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create Heatmap manually (using imshow) to avoid needing extra libraries like seaborn
    # Transpose data so: X-axis = Scenarios, Y-axis = Areas (Easier to read)
    data_values = heatmap_data.values

    # Plot the heatmap
    im = ax.imshow(
        data_values, cmap="YlGnBu", aspect="auto"
    )  # Yellow-Green-Blue colormap

    # Add Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Total Score (0-10)", rotation=-90, va="bottom")

    # Set Ticks and Labels
    # X-axis: Scenarios
    ax.set_xticks(np.arange(data_values.shape[1]))
    ax.set_xticklabels(heatmap_data.columns, fontsize=10, fontweight="bold")

    # Y-axis: Areas
    ax.set_yticks(np.arange(data_values.shape[0]))
    ax.set_yticklabels(heatmap_data.index, fontsize=12, fontweight="bold")

    # Add Score Text inside each cell
    for i in range(data_values.shape[0]):  # Rows (Areas)
        for j in range(data_values.shape[1]):  # Cols (Scenarios)
            score = data_values[i, j]
            # Choose text color based on background darkness
            text_color = "white" if score > 5 else "black"
            text = ax.text(
                j,
                i,
                f"{score:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=12,
            )

    # Final Formatting
    ax.set_title(
        "Strategic Decision Matrix: Where should we launch?",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.show()


# Generate Plot
print("\n--- Generating Decision Heatmap... ---")
plot_decision_heatmap(scenario_scores)
