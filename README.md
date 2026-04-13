# Final Year Project of The Hong Kong Polytechnic University

Drone-Delivery-and-Optimal-Path-Planning-under-Hong-Kong-Government-Drone-Control-Ordinance

## Reorganized Project Structure

```
FYP/
	docs/
		project_info/
			FYP Proposal.docx
			*.pptx
			*.pdf
		references/
			*.pdf
	data/
		raw/
			Area_choose_data.txt
			MTR Distance Data.csv
			MTR Distance Data.xlsx
			Peak Hour Delay Index.xlsx
			Simulation Data.xlsx
	src/
		analysis/
			cost/
				Cost_measurement.py
			scoring/
				scoring_model.py
		simulation/
			baseline/
				simulation.py
			monte_carlo/
				monte_carlo_simulation.py
		visualization/
			presentation/
				powerpoint_visualization_1.py
				powerpoint_visualization_2.py
		legacy/
			3.9.py
	outputs/
		figures/
			presentation/
				*.png
			simulation/
				*.png
		simulation_data/
			1_drone_100_times/
			2_drones/
			3_drones/
			4_drones/
```

## Module Classification

- analysis: scoring model and cost model scripts.
- simulation: baseline analytical simulation and Monte Carlo simulation.
- visualization: slide-ready chart/dashboard generation scripts.
- legacy: older archived script version kept for reference.
- docs: proposal, capstone admin materials, and literature references.
- data/raw: original input datasets (text/csv/xlsx).
- outputs/figures/presentation: final presentation-ready charts.
- outputs/figures/simulation: simulation summary charts.
- outputs/simulation_data: run-specific outputs grouped by drone fleet size.

## Quick Run Guide

From repository root:

```bash
source .venv/bin/activate
python FYP/src/analysis/scoring/scoring_model.py
python FYP/src/analysis/cost/Cost_measurement.py
python FYP/src/simulation/baseline/simulation.py
python FYP/src/simulation/monte_carlo/monte_carlo_simulation.py
python FYP/src/visualization/presentation/powerpoint_visualization_1.py
python FYP/src/visualization/presentation/powerpoint_visualization_2.py
```

## Note

Some simulation scripts still use a local Excel coordinate file path that should be updated to your environment before execution.
