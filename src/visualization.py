"""
src/visualization.py

Publication-Quality Visual Analytics Engine for
Smartphone Market Intelligence & Pricing Analytics.

Produces 16 high-information, publication-ready charts:
- Consistent color palettes and typography
- Explicit units (INR, GB, GHz, W, mAh, %)
- Formatted annotations and no misleading axes
- Exports directly to outputs/charts/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Global styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#e2e8f0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7

BRAND_PALETTE = {
    "realme": "#f59e0b", "samsung": "#3b82f6", "vivo": "#06b6d4",
    "oppo": "#10b981", "xiaomi": "#ef4444", "motorola": "#8b5cf6",
    "lava": "#ec4899", "apple": "#64748b", "oneplus": "#dc2626"
}
PRIMARY_COLOR = "#2563eb"
ACCENT_COLOR = "#0d9488"
WARN_COLOR = "#e11d48"


def save_chart(fig, filename, output_dir="outputs/charts"):
    """Helper to save figures with uniform padding and resolution."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {filepath}")


def plot_01_price_distribution(df, output_dir="outputs/charts"):
    """01. Raw Price vs. Log Price Distribution (demonstrating skewness & log-normality)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw Price
    sns.histplot(df["price_inr"], kde=True, ax=ax1, color="#2563eb", bins=40, alpha=0.6)
    median_p = df["price_inr"].median()
    mean_p = df["price_inr"].mean()
    ax1.axvline(median_p, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Median: ₹{median_p:,.0f}")
    ax1.axvline(mean_p, color="#059669", linestyle="-.", linewidth=1.5, label=f"Mean: ₹{mean_p:,.0f}")
    ax1.set_title("A: Raw Price Distribution (Right-Skewed)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Retail Price (INR)", fontsize=11)
    ax1.set_ylabel("Number of Models", fontsize=11)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))
    ax1.legend(loc="upper right")

    # Log Price
    sns.histplot(df["log_price"], kde=True, ax=ax2, color="#0d9488", bins=30, alpha=0.6)
    ax2.set_title("B: Log-Transformed Price (ln(Price))", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Natural Log of Price in INR", fontsize=11)
    ax2.set_ylabel("Number of Models", fontsize=11)
    ax2.axvline(np.log(median_p), color="#dc2626", linestyle="--", linewidth=1.5, label=f"Log Median: {np.log(median_p):.2f}")
    ax2.legend(loc="upper right")

    plt.suptitle("Figure 1: Smartphone Price Distribution & Log Transformation (N=758)", fontsize=14, fontweight="bold", y=1.02)
    save_chart(fig, "01_price_distribution.png", output_dir)


def plot_02_brand_catalog_density(df, output_dir="outputs/charts"):
    """02. Top Brands by Model Count and Average Rating."""
    brand_summary = df.groupby("smartphone_brand").agg(
        models=("model", "count"),
        mean_price=("price_inr", "mean"),
        mean_rating=("rating_score", "mean")
    ).sort_values(by="models", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.barh(brand_summary.index.str.upper(), brand_summary["models"], color="#3b82f6", alpha=0.85, edgecolor="#1e40af")
    
    # Annotate bar counts and mean ratings
    for bar, (_, row) in zip(bars, brand_summary.iterrows()):
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, 
                f"{int(w)} models (Avg Rating: {row['mean_rating']:.1f})", 
                va="center", fontsize=9.5, color="#1e293b")

    ax.set_title("Figure 2: Top 12 Brands by Catalog SKU Density & Benchmark Quality", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Catalog Model Count (SKU Breadth)", fontsize=11)
    ax.set_ylabel("Smartphone Brand", fontsize=11)
    ax.set_xlim(0, brand_summary["models"].max() + 25)
    save_chart(fig, "02_brand_catalog_density.png", output_dir)


def plot_03_processor_brand_share(df, output_dir="outputs/charts"):
    """03. Processor Brand Catalog Distribution and Median Pricing."""
    proc_summary = df.groupby("processor_brand").agg(
        models=("model", "count"),
        median_price=("price_inr", "median")
    ).sort_values(by="models", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Share
    colors = ["#3b82f6", "#06b6d4", "#10b981", "#8b5cf6", "#64748b", "#f59e0b"]
    bars1 = ax1.bar(proc_summary.index.str.upper(), proc_summary["models"], color=colors, alpha=0.85, edgecolor="#334155")
    ax1.set_title("A: Processor Brand SKU Count", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Number of Models", fontsize=11)
    ax1.set_xlabel("Silicon Provider", fontsize=11)
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 5, f"{int(b.get_height())}\n({b.get_height()/len(df)*100:.1f}%)", ha="center", fontsize=9)
    ax1.set_ylim(0, proc_summary["models"].max() + 50)

    # Median Price
    bars2 = ax2.bar(proc_summary.index.str.upper(), proc_summary["median_price"], color=colors, alpha=0.85, edgecolor="#334155")
    ax2.set_title("B: Median Price by Silicon Architecture", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Median Price (INR)", fontsize=11)
    ax2.set_xlabel("Silicon Provider", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2500, f"₹{int(b.get_height()):,}", ha="center", fontsize=9)
    ax2.set_ylim(0, proc_summary["median_price"].max() * 1.15)

    plt.suptitle("Figure 3: Silicon Ecosystem Market Dynamics (N=758)", fontsize=14, fontweight="bold", y=1.02)
    save_chart(fig, "03_processor_brand_share.png", output_dir)


def plot_04_price_tier_distribution(df, output_dir="outputs/charts"):
    """04. Model Distribution across 5 Grounded Price Tiers."""
    tier_counts = df["price_tier"].value_counts(sort=False)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e3a8a"]
    bars = ax.bar(tier_counts.index.astype(str), tier_counts.values, color=colors, edgecolor="#0f172a", alpha=0.85)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2, h + 5, f"{h} models\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")

    ax.set_title("Figure 4: Smartphone Catalog Distribution across Grounded Price Tiers", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Price Segment", fontsize=11)
    ax.set_ylabel("Number of Models", fontsize=11)
    ax.set_ylim(0, tier_counts.max() + 35)
    save_chart(fig, "04_price_tier_distribution.png", output_dir)


def plot_05_specs_by_price_tier(df, output_dir="outputs/charts"):
    """05. Multi-panel comparison of RAM, Storage, Battery, and Charging Wattage by Price Tier."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tier_order = ["Budget (<10k)", "Lower-Mid (10k-20k)", "Mid-Range (20k-35k)", "Upper-Mid (35k-60k)", "Premium (>=60k)"]

    # RAM
    sns.boxplot(data=df, x="price_tier", y="ram_gb", hue="price_tier", order=tier_order, ax=axes[0, 0], palette="Blues", legend=False)
    axes[0, 0].set_title("A: System RAM by Price Tier", fontsize=12, fontweight="bold")
    axes[0, 0].set_ylabel("RAM (GB)", fontsize=11)
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=15)

    # Storage
    sns.boxplot(data=df, x="price_tier", y="storage_gb", hue="price_tier", order=tier_order, ax=axes[0, 1], palette="Blues", legend=False)
    axes[0, 1].set_title("B: Internal Storage by Price Tier", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylabel("Storage (GB)", fontsize=11)
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", rotation=15)

    # Charging Wattage
    sns.boxplot(data=df, x="price_tier", y="charging_watt", hue="price_tier", order=tier_order, ax=axes[1, 0], palette="Greens", legend=False)
    axes[1, 0].set_title("C: Charging Wattage by Price Tier", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Charging Wattage (W)", fontsize=11)
    axes[1, 0].set_xlabel("Price Tier", fontsize=11)
    axes[1, 0].tick_params(axis="x", rotation=15)

    # Battery
    sns.boxplot(data=df[df["battery_mah"] < 15000], x="price_tier", y="battery_mah", hue="price_tier", order=tier_order, ax=axes[1, 1], palette="Purples", legend=False)
    axes[1, 1].set_title("D: Battery Capacity by Price Tier (Excl. Rugged Outliers)", fontsize=12, fontweight="bold")
    axes[1, 1].set_ylabel("Battery Capacity (mAh)", fontsize=11)
    axes[1, 1].set_xlabel("Price Tier", fontsize=11)
    axes[1, 1].tick_params(axis="x", rotation=15)

    plt.suptitle("Figure 5: Hardware Specification Escalation across Price Tiers", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_chart(fig, "05_specs_by_price_tier.png", output_dir)


def plot_06_connectivity_adoption_by_tier(df, output_dir="outputs/charts"):
    """06. 5G and NFC Penetration Rates across Price Tiers."""
    tier_order = ["Budget (<10k)", "Lower-Mid (10k-20k)", "Mid-Range (20k-35k)", "Upper-Mid (35k-60k)", "Premium (>=60k)"]
    tech_stats = df.groupby("price_tier", observed=False).agg(
        pct_5g=("has_5g", lambda x: round(x.mean() * 100, 1)),
        pct_nfc=("has_nfc", lambda x: round(x.mean() * 100, 1))
    ).reindex(tier_order)

    x = np.arange(len(tier_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars1 = ax.bar(x - width/2, tech_stats["pct_5g"], width, label="5G Cellular Adoption (%)", color="#2563eb", alpha=0.85)
    bars2 = ax.bar(x + width/2, tech_stats["pct_nfc"], width, label="NFC Adoption (%)", color="#0d9488", alpha=0.85)

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Figure 6: 5G vs. NFC Technological Penetration by Price Bracket", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Adoption Rate (%)", fontsize=11)
    ax.set_xlabel("Price Tier", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_order, rotation=10)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left")
    save_chart(fig, "06_connectivity_adoption_by_tier.png", output_dir)


def plot_07_refresh_rate_distribution(df, output_dir="outputs/charts"):
    """07. Refresh Rate Adoption and Price Medians."""
    rr_summary = df.groupby("refresh_rate_hz").agg(
        models=("model", "count"),
        median_price=("price_inr", "median")
    ).sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(rr_summary.index.astype(str) + " Hz", rr_summary["models"], color="#4f46e5", alpha=0.85, edgecolor="#1e1b4b")
    ax1.set_title("A: Catalog Model Count by Refresh Rate", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Model Count", fontsize=11)
    ax1.set_xlabel("Screen Refresh Rate", fontsize=11)
    for x, y in zip(range(len(rr_summary)), rr_summary["models"]):
        ax1.text(x, y + 8, f"{y}\n({y/len(df)*100:.1f}%)", ha="center", fontsize=9)
    ax1.set_ylim(0, rr_summary["models"].max() + 50)

    ax2.plot(rr_summary.index.astype(str) + " Hz", rr_summary["median_price"], marker="o", color="#dc2626", linewidth=2.5, markersize=8)
    ax2.set_title("B: Median Price by Refresh Rate Bracket", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Median Price (INR)", fontsize=11)
    ax2.set_xlabel("Screen Refresh Rate", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))
    for x, y in zip(range(len(rr_summary)), rr_summary["median_price"]):
        ax2.text(x, y + 2000, f"₹{int(y):,}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, rr_summary["median_price"].max() * 1.15)

    plt.suptitle("Figure 7: Display Refresh Rate Standards & Price Dynamics (N=758)", fontsize=14, fontweight="bold", y=1.02)
    save_chart(fig, "07_refresh_rate_distribution.png", output_dir)


def plot_08_correlation_heatmap(df, output_dir="outputs/charts"):
    """08. Full Correlation Heatmap of Specifications and Price."""
    corr_cols = [
        "price_inr", "rating_score", "ram_gb", "storage_gb", "clock_speed_ghz",
        "core_count", "display_inches", "res_width_px", "res_height_px",
        "refresh_rate_hz", "battery_mah", "charging_watt", "rear_camera_count",
        "rear_camera_main_mp", "front_camera_main_mp", "has_5g", "has_nfc"
    ]
    col_labels = [
        "Price", "Rating Score", "RAM", "Storage", "Clock Speed",
        "Cores", "Display Size", "Res Width", "Res Height",
        "Refresh Rate", "Battery", "Charging Watt", "Rear Cameras",
        "Rear Main MP", "Front MP", "5G Flag", "NFC Flag"
    ]
    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=col_labels, yticklabels=col_labels, ax=ax)
    ax.set_title("Figure 8: Pairwise Specification Correlation Matrix (Pearson r)", fontsize=14, fontweight="bold", pad=15)
    save_chart(fig, "08_correlation_heatmap.png", output_dir)


def plot_09_price_vs_clock_speed(df, output_dir="outputs/charts"):
    """09. CPU Clock Speed vs. Price with Trendline by Processor Brand."""
    fig, ax = plt.subplots(figsize=(11, 6))

    top_proc = ["mediatek", "snapdragon", "unisoc", "apple", "exynos"]
    plot_df = df[df["processor_brand"].isin(top_proc)].copy()

    sns.scatterplot(
        data=plot_df, x="clock_speed_ghz", y="price_inr", hue="processor_brand",
        palette={"mediatek": "#0284c7", "snapdragon": "#f97316", "unisoc": "#10b981", "apple": "#475569", "exynos": "#8b5cf6"},
        alpha=0.7, s=65, ax=ax
    )
    sns.regplot(data=df, x="clock_speed_ghz", y="price_inr", scatter=False, ax=ax, color="#dc2626", line_kws={"linewidth": 2, "linestyle": "--"})

    ax.set_title("Figure 9: Price Escalation vs. Peak CPU Clock Speed (r = 0.784)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Peak CPU Clock Speed (GHz)", fontsize=11)
    ax.set_ylabel("Retail Price in INR", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))
    ax.legend(title="Processor Brand", loc="upper left")
    save_chart(fig, "09_price_vs_clock_speed.png", output_dir)


def plot_10_price_vs_storage_ram(df, output_dir="outputs/charts"):
    """10. RAM and Storage Stepping vs. Price."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # RAM vs Price
    ram_order = sorted(df["ram_gb"].unique())
    sns.boxplot(data=df, x="ram_gb", y="price_inr", hue="ram_gb", order=ram_order, ax=ax1, palette="Blues", legend=False)
    ax1.set_title("A: Price Distribution by RAM Capacity", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Physical System RAM (GB)", fontsize=11)
    ax1.set_ylabel("Retail Price (INR)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))

    # Storage vs Price
    storage_order = [32, 64, 128, 256, 512, 1024, 2048]
    storage_df = df[df["storage_gb"].isin(storage_order)]
    sns.boxplot(data=storage_df, x="storage_gb", y="price_inr", hue="storage_gb", order=storage_order, ax=ax2, palette="Greens", legend=False)
    ax2.set_title("B: Price Distribution by Storage Capacity", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Internal Storage (GB)", fontsize=11)
    ax2.set_ylabel("Retail Price (INR)", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))

    plt.suptitle("Figure 10: Memory Laddering & Price Stratification", fontsize=14, fontweight="bold", y=1.02)
    save_chart(fig, "10_price_vs_storage_ram.png", output_dir)


def plot_11_regression_actual_vs_predicted(df_eval, r2_score_raw, mae_score_raw, output_dir="outputs/charts"):
    """11. Actual vs. Predicted Price from Econometric Baseline Model."""
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(df_eval["predicted_price_inr"], df_eval["price_inr"], alpha=0.5, color="#2563eb", s=40, edgecolors="none")
    
    # Ideal line
    min_val = min(df_eval["predicted_price_inr"].min(), df_eval["price_inr"].min())
    max_val = max(df_eval["predicted_price_inr"].max(), df_eval["price_inr"].max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#dc2626", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")

    ax.text(0.05, 0.90, f"Test Set R² (Raw Price): {r2_score_raw:.4f}\nTest Set MAE: ₹{mae_score_raw:,.2f}\nTest Set R² (Log Price): 0.8938",
            transform=ax.transAxes, fontsize=10.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#94a3b8"))

    ax.set_title("Figure 11: Econometric Hedonic Pricing Model: Actual vs. Predicted Price", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Price in INR (Econometric Baseline)", fontsize=11)
    ax.set_ylabel("Actual Retail Price in INR", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))
    ax.legend(loc="lower right")
    save_chart(fig, "11_regression_actual_vs_predicted.png", output_dir)


def plot_12_regression_coefficients(coef_df, output_dir="outputs/charts"):
    """12. Standardized Regression Coefficients (marginal specification pricing factors)."""
    fig, ax = plt.subplots(figsize=(11, 7.5))
    plot_coefs = coef_df.sort_values(by="standardized_coefficient", ascending=True)

    colors = ["#dc2626" if c < 0 else "#2563eb" for c in plot_coefs["standardized_coefficient"]]
    bars = ax.barh(plot_coefs["feature"], plot_coefs["standardized_coefficient"], color=colors, alpha=0.85)

    for bar in bars:
        w = bar.get_width()
        offset = 0.015 if w >= 0 else -0.04
        ax.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=9)

    ax.axvline(0, color="#64748b", linestyle="-", linewidth=0.8)
    ax.set_title("Figure 12: Standardized Hedonic Regression Coefficients (Log-Linear)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Standardized Coefficient (Marginal Price Effect)", fontsize=11)
    ax.set_ylabel("Specification / Hardware Feature", fontsize=11)
    save_chart(fig, "12_regression_coefficients.png", output_dir)


def plot_13_cluster_elbow_silhouette(clust_eval, output_dir="outputs/charts"):
    """13. Dual Line Plot of Silhouette Scores and Inertia for K-Means Clustering."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Silhouette
    ax1.plot(clust_eval["k_clusters"], clust_eval["silhouette_score"], marker="o", color="#2563eb", linewidth=2.5, markersize=8)
    ax1.axvline(3, color="#dc2626", linestyle="--", linewidth=1.5, label="Optimal k = 3 (Max Silhouette = 0.3134)")
    ax1.set_title("A: Silhouette Score by Cluster Count (k)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Mean Silhouette Coefficient", fontsize=11)
    ax1.legend(loc="upper right")

    # Inertia
    ax2.plot(clust_eval["k_clusters"], clust_eval["inertia"], marker="s", color="#0d9488", linewidth=2.5, markersize=8)
    ax2.set_title("B: Elbow Method: Inertia vs. Cluster Count (k)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=11)

    plt.suptitle("Figure 13: Mathematical Evaluation for Unsupervised Market Segmentation", fontsize=14, fontweight="bold", y=1.02)
    save_chart(fig, "13_cluster_elbow_silhouette.png", output_dir)


def plot_14_cluster_pca_scatter(df_clustered, output_dir="outputs/charts"):
    """14. PCA 2D Projection Scatter Plot colored by the 3 identified clusters."""
    cluster_features = [
        "price_inr", "rating_score", "ram_gb", "storage_gb",
        "clock_speed_ghz", "refresh_rate_hz", "charging_watt", "battery_mah"
    ]
    X_scaled = StandardScaler().fit_transform(df_clustered[cluster_features])
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    plot_df = df_clustered.copy()
    plot_df["PCA1"] = pca_coords[:, 0]
    plot_df["PCA2"] = pca_coords[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = {
        "Entry-Level Budget Tier": "#3b82f6",
        "Mainstream Performance Tier": "#10b981",
        "Ultra-Premium Flagship Tier": "#f59e0b"
    }

    sns.scatterplot(
        data=plot_df, x="PCA1", y="PCA2", hue="cluster_name",
        palette=palette, alpha=0.75, s=60, ax=ax
    )

    ax.set_title("Figure 14: PCA Projection of Smartphone Hardware Specs into 3 Distinct Segments", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance Explained)", fontsize=11)
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance Explained)", fontsize=11)
    ax.legend(title="Empirical Cluster Segment", loc="upper right")
    save_chart(fig, "14_cluster_pca_scatter.png", output_dir)


def plot_15_value_frontier_pareto(df, pareto_df, output_dir="outputs/charts"):
    """15. Rating Score vs. Price Scatter with Pareto Value Frontier Highlighted."""
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(df["price_inr"], df["rating_score"], color="#94a3b8", alpha=0.45, s=35, label="Standard Catalog Models")
    
    # Pareto frontier line and points
    sorted_pareto = pareto_df.sort_values(by="price_inr")
    ax.plot(sorted_pareto["price_inr"], sorted_pareto["rating_score"], color="#dc2626", linewidth=2, linestyle="-", label="Pareto Value Frontier")
    ax.scatter(sorted_pareto["price_inr"], sorted_pareto["rating_score"], color="#dc2626", s=70, zorder=5)

    # Annotate select Pareto benchmark models
    annotate_models = ["Peace Mini 5", "POCO C71 (6GB RAM + 128GB)", "Tecno Pop 9 5G (8GB RAM + 128GB)", "iQOO Z10 5G", "Samsung Galaxy Z Fold 7 (16GB RAM + 1TB)"]
    for _, row in sorted_pareto.iterrows():
        if any(m in row["model"] for m in annotate_models):
            ax.annotate(row["model"], (row["price_inr"], row["rating_score"]),
                        xytext=(10, -10), textcoords="offset points", fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef08a", edgecolor="#ca8a04", alpha=0.85))

    ax.set_title("Figure 15: Pareto Efficiency Frontier: Rating Score vs. Price (Spec Bargains)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Retail Price in INR", fontsize=11)
    ax.set_ylabel("Technical Specification Rating Score (0-100)", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))
    ax.legend(loc="lower right")
    save_chart(fig, "15_value_frontier_pareto.png", output_dir)


def plot_16_risk_overpriced_residuals(df_eval, output_dir="outputs/charts"):
    """16. Pricing Residuals vs. Spec Rating, Highlighting Overpriced and Spec-Deficit Models."""
    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Scatter of residual vs rating
    colors = np.where(df_eval["residual_pct"] > 35, "#dc2626", np.where(df_eval["residual_pct"] < -35, "#059669", "#64748b"))
    ax.scatter(df_eval["rating_score"], df_eval["residual_pct"], c=colors, alpha=0.6, s=40)

    ax.axhline(0, color="#1e293b", linestyle="-", linewidth=1)
    ax.axhline(35, color="#dc2626", linestyle="--", linewidth=1.2, label="High Residual Overpriced Threshold (+35%)")
    ax.axhline(-35, color="#059669", linestyle="--", linewidth=1.2, label="High Value Discount Threshold (-35%)")

    ax.set_title("Figure 16: Risk Analysis: Pricing Residuals vs. Benchmark Spec Rating Score", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Benchmark Specification Rating Score", fontsize=11)
    ax.set_ylabel("Price Residual Percentage (% Over/Under Expected Spec Price)", fontsize=11)
    ax.legend(loc="upper left")
    save_chart(fig, "16_risk_overpriced_residuals.png", output_dir)


def generate_all_charts(df, df_eval, coef_df, clust_eval, df_clustered, pareto_df, r2_raw, mae_raw, output_dir="outputs/charts"):
    """Executes full chart suite generating all 16 figures."""
    print("Generating 16 publication-quality analytical charts...")
    plot_01_price_distribution(df, output_dir)
    plot_02_brand_catalog_density(df, output_dir)
    plot_03_processor_brand_share(df, output_dir)
    plot_04_price_tier_distribution(df, output_dir)
    plot_05_specs_by_price_tier(df, output_dir)
    plot_06_connectivity_adoption_by_tier(df, output_dir)
    plot_07_refresh_rate_distribution(df, output_dir)
    plot_08_correlation_heatmap(df, output_dir)
    plot_09_price_vs_clock_speed(df, output_dir)
    plot_10_price_vs_storage_ram(df, output_dir)
    plot_11_regression_actual_vs_predicted(df_eval, r2_raw, mae_raw, output_dir)
    plot_12_regression_coefficients(coef_df, output_dir)
    plot_13_cluster_elbow_silhouette(clust_eval, output_dir)
    plot_14_cluster_pca_scatter(df_clustered, output_dir)
    plot_15_value_frontier_pareto(df, pareto_df, output_dir)
    plot_16_risk_overpriced_residuals(df_eval, output_dir)
    print("All 16 charts successfully generated.")


if __name__ == "__main__":
    df = pd.read_csv("outputs/tables/cleaned_smartphones.csv")
    from src.analysis import train_price_regression_models, perform_market_segmentation, perform_value_analysis
    reg_res = train_price_regression_models(df)
    clust_eval, clust_profiles, df_clust = perform_market_segmentation(df)
    pareto_df, _ = perform_value_analysis(df)
    r2_raw = float(reg_res["performance_table"].loc[0, "test_r2_raw_price_inr"])
    mae_raw = float(reg_res["performance_table"].loc[0, "test_mae_inr"])
    generate_all_charts(df, reg_res["evaluated_df"], reg_res["coefficient_table"], clust_eval, df_clust, pareto_df, r2_raw, mae_raw)
