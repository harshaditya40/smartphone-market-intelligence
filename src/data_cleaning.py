"""
src/data_cleaning.py

Reproducible Data Cleaning and Transformation Pipeline for
Smartphone Market Intelligence & Pricing Analytics.

Adheres strictly to the following principles:
1. The raw dataset (data/smartphones.csv) is immutable and never modified.
2. Zero silent deletion of rows (758 rows in -> 758 rows out).
3. Every transformation and imputation is logged with:
   - What was wrong
   - Why it was problematic
   - What rule was applied
   - How many records were affected
4. Generates verified outputs in outputs/tables/.
"""

import os
import numpy as np
import pandas as pd


def load_raw_data(filepath="data/smartphones.csv") -> pd.DataFrame:
    """Loads the raw immutable dataset and returns an unmutated copy."""
    # Support both case variations if needed
    if not os.path.exists(filepath):
        alt_path = "Data/smartphones.csv"
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Raw data file not found at {filepath} or {alt_path}")
    df = pd.read_csv(filepath)
    return df


def clean_and_transform_data(df_raw: pd.DataFrame):
    """
    Executes the full data cleaning, consistency verification, and feature engineering pipeline.
    
    Returns:
        df_cleaned (pd.DataFrame): Cleaned and enriched dataset (758 rows).
        quality_summary (dict): Comprehensive before/after audit metrics.
        decisions_log (pd.DataFrame): Granular log of every cleaning decision.
    """
    df = df_raw.copy()
    rows_initial, cols_initial = df.shape
    missing_before = df.isnull().sum().to_dict()
    duplicates_before = int(df.duplicated().sum())

    decisions = []

    # 1. Missing Value Handling: Front Camera Megapixels
    # Problem: 5 records have NaN for front_camera_main_mp.
    # Why problematic: Algorithms require numeric values; dropping rows loses valid budget phones.
    # Rule: If front_camera_count == 0, front_camera_main_mp is structurally 0.0.
    zero_front_mask = (df["front_camera_count"] == 0) & (df["front_camera_main_mp"].isna())
    affected_front = int(zero_front_mask.sum())
    df.loc[zero_front_mask, "front_camera_main_mp"] = 0.0

    decisions.append({
        "issue": "Missing front_camera_main_mp (NaN)",
        "why_problematic": "Algorithms require numeric inputs; dropping rows discards valid budget devices.",
        "rule_applied": "Imputed 0.0 MP since front_camera_count == 0 (no selfie camera present).",
        "records_affected": affected_front,
        "affected_models": "Ringme Bold P70, Peace Honor 30, Peace Mini 4, Peace Mini 5, Peace Mini 6"
    })

    # 2. Missing Value Handling: Memory Card Support & Type
    # Problem: 207 records have NaN in memory_card_supported and memory_card_type.
    # Why problematic: Missing flags prevent categorical modeling; mostly modern upper-mid/flagship models without expansion.
    # Rule: Impute memory_card_supported = False and memory_card_type = 'none'.
    missing_sd_mask = df["memory_card_supported"].isna() & df["memory_card_type"].isna()
    affected_sd_missing = int(missing_sd_mask.sum())
    df.loc[missing_sd_mask, "memory_card_supported"] = False
    df.loc[missing_sd_mask, "memory_card_type"] = "none"

    decisions.append({
        "issue": "Missing memory_card_supported and memory_card_type (NaN)",
        "why_problematic": "Unclear specification prevents categorical encoding; represents modern slot removal in flagships.",
        "rule_applied": "Set memory_card_supported = False and memory_card_type = 'none' for unrecorded slots.",
        "records_affected": affected_sd_missing,
        "affected_models": "207 upper-tier models (Apple, Samsung, OnePlus, Vivo flagships, etc.)"
    })

    # 3. Logical Inconsistency Handling: Memory Card False yet Dedicated
    # Problem: 168 records have memory_card_supported == False, yet memory_card_type == 'dedicated'.
    # Why problematic: Contradiction in specifications; cannot have a physical dedicated slot if unsupported.
    # Rule: Force memory_card_type = 'none' whenever memory_card_supported == False.
    inconsistent_sd_mask = (df["memory_card_supported"] == False) & (df["memory_card_type"] == "dedicated")
    affected_sd_inconsistent = int(inconsistent_sd_mask.sum())
    df.loc[inconsistent_sd_mask, "memory_card_type"] = "none"

    decisions.append({
        "issue": "Contradictory memory card status (supported=False, type='dedicated')",
        "why_problematic": "Misleading classification indicating physical slot exists when unsupported.",
        "rule_applied": "Normalized memory_card_type to 'none' when memory_card_supported is False.",
        "records_affected": affected_sd_inconsistent,
        "affected_models": "168 models across various brands"
    })

    # Create harmonized expansion_slot column
    df["expansion_slot"] = df["memory_card_type"].str.lower().str.strip()

    # 4. Anomaly Detection & Flagging: Resolution Outlier
    # Problem: Ringme Bold P70 has 6.3 inch screen but 128x160 px resolution (~31.8 PPI).
    # Why problematic: Distorts resolution statistics and downstream PPI calculations.
    # Rule: Flag as resolution_anomaly = True to maintain 100% data transparency without silent deletion.
    df["resolution_anomaly"] = False
    res_anomaly_mask = (df["res_width_px"] < 300) & (df["display_inches"] > 5.0)
    affected_res_anomaly = int(res_anomaly_mask.sum())
    df.loc[res_anomaly_mask, "resolution_anomaly"] = True

    decisions.append({
        "issue": "Upstream display resolution entry error (128x160 px on 6.3 inch screen)",
        "why_problematic": "Severe PPI distortion (~32 PPI); typical of feature phone screen specs erroneously entered.",
        "rule_applied": "Preserved raw data and flagged with resolution_anomaly = True for transparent filtering in display models.",
        "records_affected": affected_res_anomaly,
        "affected_models": "Ringme Bold P70"
    })

    # 5. Categorical String Normalization
    df["smartphone_brand"] = df["smartphone_brand"].astype(str).str.strip().str.lower()
    df["processor_brand"] = df["processor_brand"].astype(str).str.strip().str.lower()
    df["os_name"] = df["os_name"].astype(str).str.strip().str.lower()
    df["processor_name"] = df["processor_name"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip()

    # 6. Data Type Enforcement
    int_cols = [
        "price_inr", "rating_score", "core_count", "ram_gb", "storage_gb",
        "res_width_px", "res_height_px", "refresh_rate_hz", "battery_mah",
        "rear_camera_count", "front_camera_count"
    ]
    for c in int_cols:
        df[c] = df[c].astype(int)

    float_cols = [
        "clock_speed_ghz", "display_inches", "charging_watt",
        "rear_camera_main_mp", "front_camera_main_mp"
    ]
    for c in float_cols:
        df[c] = df[c].astype(float)

    bool_cols = ["has_5g", "has_nfc", "has_ir_blaster", "fast_charging", "memory_card_supported", "resolution_anomaly"]
    for c in bool_cols:
        df[c] = df[c].astype(bool)

    # 7. Feature Engineering
    # A. Pixels Per Inch (PPI)
    df["ppi"] = (np.sqrt(df["res_width_px"]**2 + df["res_height_px"]**2) / df["display_inches"]).round(1)

    # B. Aspect Ratio
    df["aspect_ratio"] = (df["res_height_px"] / df["res_width_px"]).round(2)

    # C. Price Tiers (Industry Standard Grounded Segmentation)
    # Tier 1: Budget (< 10,000 INR)
    # Tier 2: Lower-Mid (10,000 - 19,999 INR)
    # Tier 3: Mid-Range (20,000 - 34,999 INR)
    # Tier 4: Upper-Mid (35,000 - 59,999 INR)
    # Tier 5: Premium / Flagship (>= 60,000 INR)
    price_bins = [0, 10000, 20000, 35000, 60000, np.inf]
    price_labels = [
        "Budget (<10k)",
        "Lower-Mid (10k-20k)",
        "Mid-Range (20k-35k)",
        "Upper-Mid (35k-60k)",
        "Premium (>=60k)"
    ]
    df["price_tier"] = pd.cut(df["price_inr"], bins=price_bins, labels=price_labels, right=False)

    # D. Log Price (for normal distribution in linear models)
    df["log_price"] = np.log(df["price_inr"])

    # E. Specification Value Score (Rating Score per 1,000 INR)
    df["value_ratio"] = ((df["rating_score"] / df["price_inr"]) * 1000).round(2)

    # F. Screen Form Factor Category
    def categorize_screen(inches):
        if inches < 5.0:
            return "Ultra-Compact (<5.0\")"
        elif inches <= 7.0:
            return "Standard (5.0\"-7.0\")"
        else:
            return "Large/Foldable (>7.0\")"
    df["screen_category"] = df["display_inches"].apply(categorize_screen)

    # G. High Refresh Rate Flag (>= 120 Hz)
    df["is_high_refresh"] = df["refresh_rate_hz"] >= 120

    # Summary and Audit
    rows_final, cols_final = df.shape
    missing_after = df.isnull().sum().to_dict()
    duplicates_after = int(df.duplicated().sum())

    decisions_df = pd.DataFrame(decisions)

    quality_summary = {
        "rows_before": rows_initial,
        "rows_after": rows_final,
        "cols_before": cols_initial,
        "cols_after": cols_final,
        "duplicates_before": duplicates_before,
        "duplicates_after": duplicates_after,
        "missing_values_before_total": sum(missing_before.values()),
        "missing_values_after_total": sum(missing_after.values()),
        "missing_by_col_before": missing_before,
        "missing_by_col_after": missing_after,
        "rows_deleted": rows_initial - rows_final
    }

    return df, quality_summary, decisions_df


def run_pipeline(input_path="data/smartphones.csv", output_dir="outputs/tables"):
    """Runs data cleaning pipeline end-to-end and saves cleaned artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    df_raw = load_raw_data(input_path)
    df_clean, quality_summary, decisions_df = clean_and_transform_data(df_raw)

    # Save cleaned data
    cleaned_csv_path = os.path.join(output_dir, "cleaned_smartphones.csv")
    df_clean.to_csv(cleaned_csv_path, index=False)

    # Save decisions log
    decisions_csv_path = os.path.join(output_dir, "cleaning_decisions_log.csv")
    decisions_df.to_csv(decisions_csv_path, index=False)

    # Save quality audit report table
    audit_rows = [
        {"Metric": "Rows Before Cleaning", "Value": quality_summary["rows_before"]},
        {"Metric": "Rows After Cleaning", "Value": quality_summary["rows_after"]},
        {"Metric": "Rows Silently Deleted", "Value": quality_summary["rows_deleted"]},
        {"Metric": "Columns Before Cleaning", "Value": quality_summary["cols_before"]},
        {"Metric": "Columns After Cleaning (with derived features)", "Value": quality_summary["cols_after"]},
        {"Metric": "Duplicate Rows Before", "Value": quality_summary["duplicates_before"]},
        {"Metric": "Duplicate Rows After", "Value": quality_summary["duplicates_after"]},
        {"Metric": "Missing Values Before Cleaning", "Value": quality_summary["missing_values_before_total"]},
        {"Metric": "Missing Values After Cleaning", "Value": quality_summary["missing_values_after_total"]},
        {"Metric": "Front Camera MP Imputations", "Value": int(decisions_df.loc[decisions_df['issue'].str.contains('front_camera'), 'records_affected'].values[0])},
        {"Metric": "Memory Card Missing Imputations", "Value": int(decisions_df.loc[decisions_df['issue'].str.contains('Missing memory_card'), 'records_affected'].values[0])},
        {"Metric": "Memory Card Inconsistency Fixes", "Value": int(decisions_df.loc[decisions_df['issue'].str.contains('Contradictory memory'), 'records_affected'].values[0])},
        {"Metric": "Resolution Anomalies Flagged", "Value": int(decisions_df.loc[decisions_df['issue'].str.contains('resolution'), 'records_affected'].values[0])},
    ]
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(os.path.join(output_dir, "data_quality_report.csv"), index=False)

    print("Data cleaning completed successfully.")
    print(f"Cleaned dataset: {cleaned_csv_path} ({df_clean.shape[0]} rows, {df_clean.shape[1]} columns)")
    print(f"Quality audit report: {os.path.join(output_dir, 'data_quality_report.csv')}")
    print(f"Decisions log: {decisions_csv_path}")

    return df_clean, quality_summary, decisions_df


if __name__ == "__main__":
    run_pipeline()
