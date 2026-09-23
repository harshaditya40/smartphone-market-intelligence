# Smartphone Market Intelligence & Pricing Analytics
### Data-Driven Analysis of Smartphone Pricing, Specifications, Brands, and Market Segments

**Program:** IBM SkillsBuild Data Analytics with AI Academic Internship  
**Candidate:** Harsha  
**Role:** Lead Data Analytics Engineer  
**Dataset:** `data/smartphones.csv` (758 Observations, 27 Initial Variables)  
**Execution Environment:** Python 3.13 | Pandas | Scikit-Learn | Seaborn | Matplotlib | Python-Docx  

---

## 1. Project Overview & Scope Limitation

> [!IMPORTANT]
> **SCOPE NOTE & DATASET REPRESENTATION:**  
> **This project analyzes a Kaggle catalog of smartphone models and their technical and pricing attributes. The dataset does not represent actual unit sales, revenue, consumer demand, market share, or complete coverage of the Indian smartphone market.**

The smartphone marketplace in India is one of the most competitive consumer technology landscapes globally. With hundreds of actively marketed models across 25+ manufacturers, original equipment manufacturers (OEMs) and product strategists face intense margin pressure, hardware specification commoditization, and consumer decision fatigue.

### Core Industry Challenges Addressed:
1. **Opaque Pricing Mechanics:** Determining the empirical monetary associations linked with specific component upgrades (e.g., RAM steps, internal storage tiers, CPU clock speed).
2. **Catalog Saturated Corridors:** Managing severe SKU crowding in the high-volume ₹10,000–₹20,000 price band where differentiation is minimal and brand self-cannibalization risk is high.
3. **Identification of Strategic Whitespace:** Detecting underserved feature-price intersections (e.g., affordable NFC integration or accessible hyper-fast charging).

> [!WARNING]
> **CRITICAL METHODOLOGICAL GUARDRAILS:**
> - **Catalog Representation $\ne$ Market Share:** The 758 models represent active catalog listings (SKU breadth), NOT retail unit sales volume, shipment figures, or financial revenue.
> - **Absence of Consumer Demand:** The dataset contains no consumer sales velocity, unit shipment numbers, or purchase transaction counts.
> - **Correlation $\ne$ Causation:** Econometric regression coefficients reflect statistical price associations and hardware bundling patterns within the analyzed dataset, not physical component manufacturing costs.
> - **Technical Specification Benchmark:** The `rating_score` is an algorithmic hardware technical/specification rating score, NOT customer satisfaction survey ratings or consumer preference metrics.
> - **Cross-Sectional Scope:** The data captures a single cross-sectional catalog snapshot in Indian Rupees (INR) and does not model longitudinal inflation or multi-year depreciation.

---

## 2. Project Objectives

1. **Establish Grounded Catalog Baselines:** Independently calculate verified descriptive KPIs across pricing, compute, memory, display, battery, and connectivity.
2. **Map Structural Market Segments:** Profile brand catalog breadth, silicon vendor dynamics, and grounded price tiers.
3. **Quantify Pricing Drivers via Econometric Modeling:** Develop an interpretable hedonic regression model to isolate the marginal price association of each technical specification.
4. **Perform Objective Market Clustering:** Evaluate whether unsupervised clustering is mathematically justified and segment the catalog based on hardware archetypes.
5. **Map Value Frontiers & Market Risks:** Identify Pareto-optimal value-frontier candidates, specification anomalies, and potential technology-positioning risks.
6. **Deploy an AI Executive Analyst with Validation:** Implement a Gemini-powered executive analyst backed by a deterministic fallback and a validation audit layer.
7. **Formulate Traceable OEM Considerations:** Translate analytical findings into concrete, data-informed considerations and potential actions.

---

## 3. Project Directory Structure

```text
Smartphone_Market_Intelligence/
│
├── data/
│   └── smartphones.csv                             # Raw immutable source dataset (758 rows, 27 cols)
│
├── notebooks/
│   └── smartphone_market_intelligence.ipynb        # Master analytical notebook (63 cells, 23 executed code cells)
│
├── src/
│   ├── __init__.py                                 # Package initialization
│   ├── data_cleaning.py                            # Reproducible cleaning, validation & feature pipeline
│   ├── analysis.py                                 # Statistical analysis, econometric regression & clustering
│   ├── visualization.py                            # 16 publication-quality analytical visualizations
│   └── ai_executive_analyst.py                     # Optional AI Executive Analyst & claim validation engine
│
├── outputs/
│   ├── charts/                                     # 17 exported publication-ready figures (PNG, 300 DPI)
│   ├── tables/                                     # Cleaned data, audit logs, KPIs, summaries & validation CSVs
│   └── models/
│       └── price_regression_model.joblib           # Trained econometric regression pipeline
│
├── reports/
│   └── Harsha_Smartphone_Market_Intelligence_ProjectReport.docx # Comprehensive 24-section Word project report
│
├── Harsha_Smartphone_Market_Intelligence.ipynb     # Root submission copy of master notebook
├── Harsha_Smartphone_Market_Intelligence_ProjectReport.docx # Root submission copy of project report
├── SUBMISSION_CHECKLIST.md                         # Final submission verification checklist
├── PROJECT_PLAN.md                                 # Foundational reconnaissance & planning blueprint
├── README.md                                       # Comprehensive repository documentation & replication guide
├── requirements.txt                                # Pinned project dependencies
└── .gitignore                                      # Project exclusion rules
```

---

## 4. Technologies & Libraries Used

| Technology / Library | Version | Role in Project |
|---|---|---|
| **Python** | `3.13.7` | Primary programming and analytical execution environment |
| **Pandas** | `3.0.3` | High-performance tabular data manipulation and aggregation |
| **NumPy** | `2.4.6` | Vectorized numerical computing and log-transformations |
| **Scipy** | `1.17.1` | Scientific distribution analysis and statistical testing |
| **Scikit-Learn** | `1.9.0` | Econometric regression (OLS, Ridge, Lasso) and K-Means clustering |
| **Matplotlib** | `3.11.0` | Publication-quality chart rendering and layout orchestration |
| **Seaborn** | `0.13.2` | Statistical data visualizations and correlation heatmaps |
| **Python-Docx** | `1.2.0` | Automated 24-section formal academic Word report generation |
| **Joblib** | `1.5.3` | Serialized model pipeline persistence |
| **IPython / ipykernel**| `7.3.0` | Standalone Jupyter execution and output generation |

---

## 5. Data Cleaning & Integrity Framework (`src/data_cleaning.py`)

The raw dataset (`data/smartphones.csv`) is preserved as immutable. Four explicit data cleaning rules were logged and applied without silently deleting any records (**758 rows in $\rightarrow$ 758 rows out**):

1. **Front Camera Megapixels:** 5 records had missing values (`NaN`). Verified `front_camera_count == 0` (no selfie camera present on ultra-budget devices); imputed `0.0 MP` (Ringme Bold P70, Peace Honor 30, Peace Mini 4, 5, 6).
2. **Expansion Slot Nulls:** 207 records lacked memory card entries, predominantly modern upper-tier phones without expansion. Imputed `supported = False` and `type = 'none'`.
3. **Memory Card Inconsistency:** In 168 records, `supported == False` yet `type == 'dedicated'`. Normalized `type = 'none'` when unsupported.
4. **Upstream Display Resolution Typo:** `Ringme Bold P70` records a 6.3" screen with 128x160 px (~32 PPI). Flagged with `resolution_anomaly = True` to enable transparent filtering in display models without data loss.

**Derived Features Added:**
- `ppi` (Pixels Per Inch): $\frac{\sqrt{W^2 + H^2}}{\text{Inches}}$
- `price_tier`: Budget (<₹10k), Lower-Mid (₹10k–₹20k), Mid-Range (₹20k–₹35k), Upper-Mid (₹35k–₹60k), Premium ($\ge$₹60k)
- `log_price`: $\ln(\text{Price})$ to normalize extreme right-skewness (+3.04 $\rightarrow$ +0.16)
- `value_ratio`: Spec score return per ₹1,000 spent ($(\text{Rating} / \text{Price}) \times 1000$)
- `screen_category`: Ultra-Compact (<5.0"), Standard (5.0"–7.0"), Large/Foldable (>7.0")
- `is_high_refresh`: Boolean flag for $\ge 120\text{Hz}$ display refresh rate

---

## 6. Key Verified Analytical Findings

### 6.1 Verified Market KPIs
- **Catalog Size:** 758 unique smartphone models across 26 active brands.
- **Price Benchmarks:** Median retail price is **₹18,999.00**; mean price is **₹29,424.09**; IQR is **₹18,132.75**. Prices range from ₹4,040 to ₹2,29,900.
- **Hardware Baseline Standards:** Median device features **8GB RAM, 128GB Storage, 5,000 mAh Battery, 44W Charging, and 120Hz Refresh Rate**.
- **Connectivity Adoption:** 5G penetration is **86.68%** overall; Fast Charging is near-universal at **98.42%**; NFC penetration is **37.20%**.

### 6.2 Market Structure & Price Tiers
- **Mass-Market Concentration:** 63.06% of models cluster in the ₹10,000–₹35,000 corridor (277 in Lower-Mid, 201 in Mid-Range).
- **Brand Catalog Density:** Realme (134 models, 17.7%), Samsung (104 models, 13.7%), and Vivo (94 models, 12.4%) represent 43.8% of all catalog offerings.
- **Silicon Architecture:** MediaTek powers 51.2% of catalog listings (388 models, anchored by Dimensity 6300), followed by Qualcomm Snapdragon at 28.9% (219 models), Unisoc at 9.0%, Samsung Exynos at 5.8%, Apple at 3.7%, and Google Tensor at 1.5%.

### 6.3 Driver Analysis & Econometric Pricing Regression
- **Primary Pricing Associates:** CPU clock speed ($r = 0.784$) and internal storage ($r = 0.754$) exhibit the strongest direct associations with price.
- **Hedonic Baseline Model Performance:** Multiple Linear Regression on $\ln(\text{Price})$ achieves:
  - **The model explains approximately 89.24% of the variation in log-transformed listed prices within the analyzed dataset (Test $R^2$: 0.8924).**
  - **Test $R^2$ (Raw Price in INR): 0.8184**
  - **Test MAE: ₹5,321.21** | **Test RMSE: ₹10,657.27**
- **Top Marginal Price Associations (Standardized Coefficients within Model):** CPU clock speed (+0.258), Internal storage (+0.244), iOS ecosystem premium (+0.218), NFC presence (+0.142), and System RAM (+0.126).

### 6.4 Unsupervised Market Segmentation (Clustering)
K-Means clustering across standardized technical and pricing variables identified **$k = 3$** as the mathematically optimal grouping (highest Silhouette Score = **0.3134**):
1. **Entry-Level Budget Tier (35.4%, 268 models):** Median Price ₹10,295 | Rating 71.0 | 4GB RAM | 18W Charging | 5,000 mAh Battery.
2. **Mainstream Performance Tier (53.6%, 406 models):** Median Price ₹22,999 | Rating 82.0 | 8GB RAM | 50W Charging | 120Hz Display | 98% 5G.
3. **Ultra-Premium Flagship Tier (11.1%, 84 models):** Median Price ₹89,999 | Rating 88.0 | 12GB RAM | 256GB Storage | 100% 5G/NFC.

### 6.5 Value Frontier (Non-Dominated Observations)
Using multi-criteria Pareto optimization, **24 value-frontier candidate models** define the non-dominated Pareto efficiency frontier under the selected price/rating criteria (offering the highest technical/specification rating score for any given budget). Key frontier benchmarks include `Tecno Pop 9 5G` (₹9,399, rating 73), `POCO C71` (₹6,798, rating 63), and `iQOO Z10 5G` (₹20,998, rating 87 with 7,300 mAh battery).

---

## 7. AI Executive Analyst & Claim Validation (`src/ai_executive_analyst.py`)

In accordance with IBM SkillsBuild training standards, an AI Executive Analyst module is implemented:
- **Workflow:** Python calculates verified metrics $\rightarrow$ Structured metrics package $\rightarrow$ AI Executive Analyst $\rightarrow$ Claim Validation Engine.
- **Optional API Integration:** If `GEMINI_API_KEY` is present in the environment, the engine connects to Google Gemini via HTTPS REST. If unavailable, it seamlessly falls back to a deterministic verified synthesis without crashing, explicitly labeled:
  `"Mode: Deterministic fallback based on verified project metrics."`
- **Validation Engine:** Automatically audits executive briefings against `outputs/tables/metrics_validation.csv` for unsupported figures, market-share claims, sales volume claims, and causal language, exporting `outputs/tables/ai_validation_report.csv`.

---

## 8. Strategic Risks, Opportunities & Action Framework

### 3 Key Market Risks
1. **Potential 4G Technology-Positioning Risk:** 11 models priced between ₹10,000 and ₹14,850 in the Lower-Mid segment lack 5G connectivity, representing potential positioning exposure against the 96.0% 5G catalog baseline in that tier (and 98.2% across models ₹12,000 and above).
2. **Lower-Mid Tier Overcrowding:** 277 models (36.5% of total catalog) crowd into the ₹10k–₹20k corridor, with 117 models sharing identical MediaTek Dimensity 6300 silicon, creating intense specification commoditization and potential catalog self-cannibalization risk.
3. **Hedonic Overpricing Vulnerability:** Multiple legacy models carry prices >35% higher than predicted by their hardware specifications while delivering sub-80 technical/specification rating scores.

### 3 Data-Indicated Opportunities
1. **Entry-Tier NFC Whitespace:** Only 7.9% of sub-₹15,000 phones feature NFC; integrating NFC in select ₹12k–₹15k devices provides a potential contactless payment differentiator for urban positioning.
2. **Hyper-Fast Charging Democratization:** Only 4.6% of sub-₹15k models feature $\ge 67\text{W}$ charging (median is 18W–33W), offering an immediate specification hook.
3. **Budget 5G + 120Hz Sweet Spot:** Only 14 models combine both 5G connectivity and a 120Hz display under ₹12,000.

### 5 Data-Informed Areas for Evaluation & Potential Actions
1. **5G Baseline Evaluation:** Evaluate 5G as a baseline specification for products above ₹12,000, given the observed 5G representation in the analyzed catalog.
2. **Targeted NFC Evaluation:** Evaluate selective NFC inclusion in the ₹12,000–₹16,000 corridor for urban contactless utility based on observed whitespace.
3. **Lower-Mid Portfolio Rationalization:** Evaluate portfolio differentiation and SKU rationalization in the crowded ₹10,000–₹20,000 bracket to mitigate catalog cannibalization.
4. **Structured Storage Step-Pricing:** Consider structured storage upgrade pricing intervals of ₹2,000 to ₹3,500 based on econometric storage coefficients.
5. **Charging Speed Evaluation:** Evaluate a 45W charging baseline for products above ₹18,000 based on observed charging characteristics in the analyzed catalog.

---

## 9. How to Reproduce

### 9.1 Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd Smartphone_Market_Intelligence

# Install pinned dependencies
python -m pip install -r requirements.txt
```

### 9.2 Optional: Set Gemini API Key
```bash
# Optional: Set Gemini API Key for live AI Analyst generation
# (If omitted, the project automatically uses the deterministic verified fallback)
$env:GEMINI_API_KEY="your-gemini-api-key"   # PowerShell
# or
export GEMINI_API_KEY="your-gemini-api-key" # Bash
```

### 9.3 Run the Pipeline End-to-End
```bash
# Step 1: Run data cleaning and quality audit
python -m src.data_cleaning

# Step 2: Run analytics, regression, clustering, and export all tables
python -m src.analysis

# Step 3: Generate all 16 publication-quality charts + Executive Dashboard
python -m src.visualization

# Step 4: Run AI Executive Analyst & claim validation engine
python -m src.ai_executive_analyst
```

### 9.4 Notebook Execution
Open and run either:
- `Harsha_Smartphone_Market_Intelligence.ipynb` (root submission file)
- `notebooks/smartphone_market_intelligence.ipynb`

Both notebooks contain 63 cells (23 executed code cells) with pre-rendered tables, outputs, and embedded figures.

### 9.5 Word Report Generation
```bash
# Generates the 24-section formal academic project report
python scratch/generate_docx_report.py
```
Output created: `Harsha_Smartphone_Market_Intelligence_ProjectReport.docx` and `reports/Harsha_Smartphone_Market_Intelligence_ProjectReport.docx`.
