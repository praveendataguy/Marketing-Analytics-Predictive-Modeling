# Marketing Analytics: Customer Behavior, Campaign Efficacy & Predictive Modeling

An end-to-end data engineering, statistical analysis, and machine learning project designed to optimize marketing spend, uncover regional purchasing behaviors, and build predictive models for customer acquisition channels.

---

## 📊 Business Objective
A company's marketing department needs to transition from generic, broad-market targeting to data-driven, localized campaigns. This project builds a robust data pipeline to:
1.  **Sanitize & Impute** messy historical customer profile data.
2.  **Analyze Campaign ROI** to understand which channels (Web, Catalog, Store, Deals) perform best globally and regionally.
3.  **Evaluate Product Popularity** to drive high-margin inventory decisions.
4.  **Predict Store Foot Traffic** by isolating statistical drivers behind in-store purchase frequency.

---

## 🛠️ Tech Stack & Architecture
*   **Data Wrangling & Pipeline Construction:** `Pandas`, `NumPy`
*   **Feature Engineering & Statistical Testing:** `SciPy (Stats)`, `F-Regression`
*   **Machine Learning (Regression & Feature Selection):** `Scikit-Learn` (`Pipeline`, `KNNImputer`, `StandardScaler`, `Lasso`, `Ridge`, `RFE`)
*   **Data Visualization:** `Matplotlib`, `Seaborn`

## 📈 Methodology & Key Findings

### 1. Robust Data Sanitization & Pipeline Integrity
*   Automated extraction of financial strings into standard floating-point metrics.
*   Constructed a leakage-free `scikit-learn Pipeline` utilizing `KNNImputer` ($k=5$) and `StandardScaler` to ensure clean downstream modeling.
*   Handled extreme anomalies logically (e.g., stripping unrealistic age indicators > 120 and flattening extreme income distribution skews).

### 2. Campaign Conversion Benchmarks
*   **The Insights:** Clear regional disparities were found in campaign uptake. While certain smaller micro-markets showed volatile spikes, high-volume regions demonstrated steady conversion floors. 
*   **Product Performance:** A deep dive into overall expenditure revealed that **Wine** and **Meat** accounted for the vast majority of total customer wallets, outperforming fruits, sweets, and gold significantly.

### 3. Feature Selection & Driver Analysis for Store Purchases
To understand what motivates a customer to make a purchase physically in-store versus online, three concurrent statistical and machine learning methodologies were deployed to rank feature importance:

*   **Regularization (LASSO & Ridge):** Used to penalize weak, collinear variables and select parsimonious features.
*   **Univariate Statistics (F-Test):** Calculated exact $p$-values to evaluate independent linear relationships.
*   **Recursive Feature Elimination (RFE):** Utilized a `RandomForestRegressor` ensemble to capture complex non-linear interactions.

Rank            Feature
   --------------------------------------
    1              NumWebVisitsMonth
    2              Income
    3              Days since last purchase
    4              MntWines
    5              NumCatalogPurchases

*Takeaway:* Web engagement metrics and income stability consistently rank as the top predictors driving high-volume brick-and-mortar store interactions.