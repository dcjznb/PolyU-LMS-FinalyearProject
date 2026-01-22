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
# PHASE 2: SCORING LOGIC (WEIGHTED FOR PROFITABILITY)
# ==========================================


def calculate_scores(data_dict):
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    scores = pd.DataFrame()
    scores["Area"] = df["Area"]

    # ---------------------------------------------------------
    # 1. Commercial Demand Score (30% Weight)
    # ---------------------------------------------------------
    # Logic: Higher Density = High Volume = Lower Fixed Cost per Order.
    d = df["Population_Density"]
    scores["Demand_Score"] = (d - d.min()) / (d.max() - d.min()) * 10

    # ---------------------------------------------------------
    # 2. Necessity Score (20% Weight)
    # ---------------------------------------------------------
    # Logic: High Congestion = Premium Pricing Power.
    c = df["Congestion_Index"]
    scores["Necessity_Score"] = (c - c.min()) / (c.max() - c.min()) * 10

    # ---------------------------------------------------------
    # 3. Efficiency Score (50% Weight - CRITICAL)
    # ---------------------------------------------------------
    # Logic: The core of cost-saving. Replacing a long, winding truck route
    # with a short drone flight generates the highest profit margin.

    # Calculate Tortuosity (Driving / Linear)
    tortuosity = df["Driving_Distance"] / (df["Linear_Distance"] + 0.01)

    # Apply Range Penalty (If > 5km, score drops significantly)
    range_penalty = np.where(df["Linear_Distance"] > 5.0, 0.2, 1.0)

    # Calculate and Normalize
    raw_efficiency = tortuosity * range_penalty
    scores["Efficiency_Score"] = (
        (raw_efficiency - raw_efficiency.min())
        / (raw_efficiency.max() - raw_efficiency.min())
        * 10
    )

    # ---------------------------------------------------------
    # 4. Total Weighted Score
    # ---------------------------------------------------------
    # Formula: Efficiency(0.5) + Demand(0.3) + Necessity(0.2)
    # This reflects a business model focused on Cost Efficiency and Profitability.
    scores["Total_Score"] = (
        scores["Efficiency_Score"] * 0.5
        + scores["Demand_Score"] * 0.3
        + scores["Necessity_Score"] * 0.2
    )

    return scores


# ==========================================
# EXECUTION
# ==========================================

# Calculate final scores with Weighted Logic
final_scores = calculate_scores(raw_data)

# Print the results
print("\n--- Phase 2: Calculated Scores (Weighted: Eff 50%, Dem 30%, Nec 20%) ---")
print(final_scores.round(2))

# ==========================================
# PHASE 3: VISUALIZATION (STACKED BAR CHART)
# ==========================================


def plot_stacked_bar(scores_df):
    # 1. Prepare Data: Calculate Weighted Contributions
    # -------------------------------------------------
    # We want to show HOW the total score is built.
    # So we don't plot the raw 0-10 score, we plot the "Weighted Points".

    areas = scores_df["Area"]

    # Calculate the height of each segment based on weights
    # Efficiency (50%), Demand (30%), Necessity (20%)
    eff_contrib = scores_df["Efficiency_Score"] * 0.5
    dem_contrib = scores_df["Demand_Score"] * 0.3
    nec_contrib = scores_df["Necessity_Score"] * 0.2

    # 2. Setup Plot
    # -------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar width
    width = 0.5
    x = np.arange(len(areas))

    # 3. Plot Stacked Bars
    # --------------------
    # Bottom layer: Efficiency (The most important foundation)
    p1 = ax.bar(
        x, eff_contrib, width, label="Cost Efficiency (50%)", color="#1f77b4", alpha=0.9
    )

    # Middle layer: Demand (Stacked on top of Efficiency)
    p2 = ax.bar(
        x,
        dem_contrib,
        width,
        bottom=eff_contrib,
        label="Commercial Demand (30%)",
        color="#ff7f0e",
        alpha=0.9,
    )

    # Top layer: Necessity (Stacked on top of previous two)
    p3 = ax.bar(
        x,
        nec_contrib,
        width,
        bottom=eff_contrib + dem_contrib,
        label="Pain Point Necessity (20%)",
        color="#d62728",
        alpha=0.9,
    )

    # 4. Add Labels & Formatting
    # --------------------------
    ax.set_ylabel("Total Weighted Score (0-10)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Site Selection Ranking: Profitability Focus",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(areas, fontsize=11, rotation=0)
    ax.set_ylim(0, 11)  # Leave some space for text on top

    # Add Legend
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Evaluation Criteria")

    # 5. Add Value Labels (The Total Score) on top of bars
    # ----------------------------------------------------
    total_scores = scores_df["Total_Score"]
    for i, v in enumerate(total_scores):
        ax.text(
            i,
            v + 0.2,
            str(round(v, 2)),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add grid for easier reading
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Adjust layout to fit legend
    plt.tight_layout()
    plt.show()


# ==========================================
# EXECUTION
# ==========================================

print("\n--- Phase 3: Generating Stacked Bar Chart... ---")
plot_stacked_bar(final_scores)
