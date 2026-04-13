import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe


# ==========================================
# Waterfall chart drawing function
# ==========================================
def draw_waterfall(ax, categories, values, title, bar_color, total_color):
    """Draw a single waterfall chart."""
    # Compute cumulative offsets
    cumulative = np.cumsum([0] + values[:-1])
    y_span = max(cumulative[-1] + values[-1], 1)

    # Color setup (final total bar uses a different color)
    colors = [bar_color] * len(values)
    colors[-1] = total_color

    # Draw bars (using bottom to create the waterfall float effect)
    bars = ax.bar(
        categories, values, bottom=cumulative, color=colors, edgecolor="white"
    )

    # Add connector lines
    for i in range(1, len(values)):
        ax.plot(
            [i - 1 - 0.4, i + 0.4],
            [cumulative[i], cumulative[i]],
            color="gray",
            linestyle="--",
            linewidth=1,
        )

    # Add value labels on bars
    for i, bar in enumerate(bars):
        yval = cumulative[i] + values[i] / 2  # Place text at bar center by default
        if i == len(values) - 1:  # For the final total bar, place text above
            yval = cumulative[i] + values[i] + y_span * 0.012
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{int(values[i]):,} HKD\n(CPD: {values[i]/10500:.2f})",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                color="black",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.9,
                    "pad": 2,
                    "edgecolor": "none",
                },
            )
        else:
            # For short bars, move labels above to avoid visual overlap
            is_small_bar = values[i] < y_span * 0.06
            if is_small_bar:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    cumulative[i] + values[i] + y_span * 0.006,
                    f"+{int(values[i]):,}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                    color="black",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.9,
                        "pad": 1.5,
                        "edgecolor": "none",
                    },
                )
            else:
                text_obj = ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval,
                    f"+{int(values[i]):,}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )
                text_obj.set_path_effects(
                    [
                        pe.Stroke(linewidth=1.2, foreground="black", alpha=0.35),
                        pe.Normal(),
                    ]
                )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=15)


# ==========================================
# 1. Data preparation
# ==========================================

# Traditional truck cost breakdown
truck_cats = [
    "Labor\n(4 Drivers)",
    "Fuel\n(850L)",
    "Tolls/Parking\n(4 Trucks)",
    "Total\nOpEx",
]
truck_vals = [
    104216,
    24880,
    24000,
    153096,
]  # Final value must equal the sum of prior items

# Drone cost breakdown
drone_cats = [
    "Labor\n(1 Operator)",
    "Battery\n(1050 Flights)",
    "Maintenance",
    "MTR/Energy\n(Share+Power)",
    "Total\nOpEx",
]
drone_vals = [
    20800,
    16695,
    5692,
    3580,
    46767,
]  # Merge MTR fee and electricity into one item to keep the chart readable

# ==========================================
# 2. Draw side-by-side waterfall charts
# ==========================================

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6), sharey=True
)  # sharey=True forces the same y-scale for direct visual comparison

# Draw truck waterfall chart (left)
draw_waterfall(
    ax1,
    truck_cats,
    truck_vals,
    title="Model A: Trucking Cost Calculation",
    bar_color="#d73027",
    total_color="#8b0000",
)

# Draw drone waterfall chart (right)
draw_waterfall(
    ax2,
    drone_cats,
    drone_vals,
    title="Model B: Drone Cost Calculation",
    bar_color="#3182bd",
    total_color="#08519c",
)

# Add baseline note at the bottom
plt.figtext(
    0.5,
    0.01,
    "Calculation Baseline: 10,500 Parcels / Month (Over 25 Operating Days)",
    ha="center",
    fontsize=12,
    fontweight="bold",
    bbox={"facecolor": "#ffffcc", "alpha": 0.5, "pad": 5},
)

plt.tight_layout(rect=(0, 0.05, 1, 1))  # Reserve space for bottom note
plt.savefig(
    "Cost_Calculation_Waterfall.png", dpi=300, transparent=False, facecolor="white"
)
plt.show()

print("Successfully generated waterfall chart: Cost_Calculation_Waterfall.png")
