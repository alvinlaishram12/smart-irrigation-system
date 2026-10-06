"""
eda.py
------
Exploratory data analysis on the simulated irrigation dataset.
Generates summary stats and saves diagnostic plots to notebooks/figures/.

Usage:
    python src/eda.py --data data/irrigation_data.csv --out notebooks/figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)


def main():
    parser = argparse.ArgumentParser(description="Run EDA on the irrigation dataset.")
    parser.add_argument("--data", type=str, default="data/irrigation_data.csv")
    parser.add_argument("--out", type=str, default="notebooks/figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)

    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)
    print("\nMISSING VALUES:\n", df.isna().sum())
    print("\nCLASS BALANCE (irrigate):\n", df["irrigate"].value_counts(normalize=True).round(3))
    print("\nNUMERIC SUMMARY:\n", df.describe().round(2).to_string())

    numeric_cols = ["soil_moisture_pct", "temperature_c", "humidity_pct",
                     "rainfall_mm", "rainfall_forecast_next_day_mm"]

    # 1. Distributions of each numeric feature, split by irrigate label
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(data=df, x=col, hue="irrigate", bins=40, kde=True,
                     stat="density", common_norm=False, ax=ax, palette=["#4472C4", "#C00000"])
        ax.set_title(col)
    axes.flat[-1].axis("off")
    plt.tight_layout()
    plt.savefig(f"{args.out}/feature_distributions.png", dpi=150)
    plt.close()

    # 2. Correlation heatmap (numeric features + label)
    corr_cols = numeric_cols + ["irrigate", "water_volume_liters_m2"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{args.out}/correlation_heatmap.png", dpi=150)
    plt.close()
    print("\nCORRELATION WITH TARGET (irrigate):\n", corr["irrigate"].sort_values(ascending=False).round(3))

    # 3. Soil moisture by irrigate decision (boxplot)
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="irrigate", y="soil_moisture_pct", palette=["#4472C4", "#C00000"])
    plt.xticks([0, 1], ["No Irrigation", "Irrigate"])
    plt.title("Soil Moisture vs. Irrigation Decision")
    plt.tight_layout()
    plt.savefig(f"{args.out}/moisture_vs_irrigate.png", dpi=150)
    plt.close()

    # 4. Irrigation rate by crop type and growth stage
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    df.groupby("crop_type")["irrigate"].mean().sort_values().plot.barh(ax=axes[0], color="#4472C4")
    axes[0].set_title("Irrigation Rate by Crop Type")
    axes[0].set_xlabel("Proportion of Days Irrigated")

    stage_order = ["Seedling", "Vegetative", "Flowering", "Maturity"]
    df.groupby("growth_stage")["irrigate"].mean().reindex(stage_order).plot.bar(ax=axes[1], color="#2E5395")
    axes[1].set_title("Irrigation Rate by Growth Stage")
    axes[1].set_ylabel("Proportion of Days Irrigated")
    plt.tight_layout()
    plt.savefig(f"{args.out}/irrigation_rate_by_category.png", dpi=150)
    plt.close()

    # 5. Soil moisture time series sample (first field, first 120 days)
    sample = df[df["field_id"] == df["field_id"].iloc[0]].head(120)
    plt.figure(figsize=(12, 4))
    plt.plot(sample["day_index"], sample["soil_moisture_pct"], color="#2E5395", label="Soil Moisture %")
    irrigated_days = sample[sample["irrigate"] == 1]
    plt.scatter(irrigated_days["day_index"], irrigated_days["soil_moisture_pct"],
                color="#C00000", zorder=5, label="Irrigation Event", s=25)
    plt.title(f"Soil Moisture Over Time — {sample['field_id'].iloc[0]}")
    plt.xlabel("Day")
    plt.ylabel("Soil Moisture (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out}/moisture_timeseries_sample.png", dpi=150)
    plt.close()

    print(f"\nSaved 5 figures to {args.out}/")


if __name__ == "__main__":
    main()
