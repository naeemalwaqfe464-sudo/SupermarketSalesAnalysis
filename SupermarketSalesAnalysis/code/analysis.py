
"""
Supermarket Sales Analysis

"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "supermarket_sales.csv"

OUT_RESULTS = ROOT / "results"
OUT_CHARTS = OUT_RESULTS / "charts"
OUT_TABLES = OUT_RESULTS / "tables"


def ensure_dirs():
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_CHARTS.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")
    return pd.read_csv(DATA)


def clean_data(df):
    # تحويل التاريخ لو موجود
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # تعبئة القيم الناقصة
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")

    return df


def save_table(df, filename):
    df.to_csv(OUT_TABLES / filename, index=False)


def save_inspection_files(df):
    df.head(10).to_csv(OUT_RESULTS / "head.csv", index=False)

    pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str)
    }).to_csv(OUT_RESULTS / "info.csv", index=False)

    df.describe(include="all").to_csv(OUT_RESULTS / "describe.csv")


def frequency_bars(df):
    cols = ["Branch", "Customer type", "Gender", "Payment"]
    for col in cols:
        if col in df.columns:
            plt.figure()
            df[col].value_counts().plot(kind="bar")
            plt.title(f"Frequency of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(OUT_CHARTS / f"bar_{col.replace(' ', '_')}.png", dpi=200)
            plt.close()


def branch_stats(df):
    if "Branch" in df.columns and "Total" in df.columns:
        stats = df.groupby("Branch")["Total"].agg(["mean", "median", "min", "max"]).reset_index()
        save_table(stats, "branch_total_stats.csv")


def sales_and_rating_over_time(df):
    if "Date" in df.columns and "Total" in df.columns:
        daily_sales = df.groupby("Date")["Total"].sum().reset_index().sort_values("Date")

        plt.figure()
        plt.plot(daily_sales["Date"], daily_sales["Total"])
        plt.title("Sales Over Time (Daily Total)")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_CHARTS / "line_sales_over_time.png", dpi=200)
        plt.close()

        save_table(daily_sales, "daily_sales.csv")

    if "Date" in df.columns and "Rating" in df.columns:
        daily_rating = df.groupby("Date")["Rating"].mean().reset_index().sort_values("Date")

        plt.figure()
        plt.plot(daily_rating["Date"], daily_rating["Rating"])
        plt.title("Rating Trend Over Time (Daily Avg)")
        plt.xlabel("Date")
        plt.ylabel("Average Rating")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_CHARTS / "line_rating_over_time.png", dpi=200)
        plt.close()

        save_table(daily_rating, "daily_rating.csv")


def scatter_sales_vs_rating(df):
    if "Total" in df.columns and "Rating" in df.columns:
        plt.figure()
        sns.scatterplot(x=df["Total"], y=df["Rating"])
        plt.title("Sales vs Rating")
        plt.tight_layout()
        plt.savefig(OUT_CHARTS / "scatter_sales_vs_rating.png", dpi=200)
        plt.close()


def correlation_heatmap(df):
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(OUT_CHARTS / "heatmap_corr.png", dpi=200)
        plt.close()

        corr.reset_index().to_csv(OUT_TABLES / "correlation_matrix.csv", index=False)


def boxplot_gross_income(df):
    if "gross income" in df.columns and "Product line" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Product line", y="gross income", data=df)
        plt.title("Gross Income by Product line")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(OUT_CHARTS / "box_gross_income_product_line.png", dpi=200)
        plt.close()


def advanced_answers(df):
    ans = {}

    # Q1: أعلى فرع من ناحية الإيرادات
    if "Branch" in df.columns and "Total" in df.columns:
        branch_rev = df.groupby("Branch")["Total"].sum().sort_values(ascending=False)
        ans["Q1"] = f"Highest revenue branch: {branch_rev.index[0]} (Total={branch_rev.iloc[0]:.2f})."
        save_table(branch_rev.reset_index(name="Revenue"), "branch_revenue.csv")

    # Q2: هل الأعضاء يصرفون أكثر؟
    if "Customer type" in df.columns and "Total" in df.columns:
        spend = df.groupby("Customer type")["Total"].mean().sort_values(ascending=False)
        ans["Q2"] = f"Avg spend by customer type: {spend.to_dict()}."
        save_table(spend.reset_index(name="AvgTotal"), "customer_type_avg_spend.csv")

    # Q3: أكثر طريقة دفع استخدام
    if "Payment" in df.columns:
        pay = df["Payment"].value_counts()
        ans["Q3"] = f"Most used payment: {pay.index[0]} (Count={pay.iloc[0]})."
        save_table(pay.reset_index(name="Count").rename(columns={"index": "Payment"}), "payment_usage.csv")

    # Q4: أعلى Product line بالتقييم
    if "Product line" in df.columns and "Rating" in df.columns:
        pr = df.groupby("Product line")["Rating"].mean().sort_values(ascending=False)
        ans["Q4"] = f"Highest avg rating product line: {pr.index[0]} (AvgRating={pr.iloc[0]:.2f})."
        save_table(pr.reset_index(name="AvgRating"), "productline_avg_rating.csv")

    # Q5: علاقة السعر بالكمية
    if "Unit price" in df.columns and "Quantity" in df.columns:
        corr = df[["Unit price", "Quantity"]].corr().iloc[0, 1]
        ans["Q5"] = f"Correlation(Unit price, Quantity) = {corr:.3f}."

    return ans


def main():
    ensure_dirs()

    df = load_data()
    save_inspection_files(df)

    df = clean_data(df)

    frequency_bars(df)
    branch_stats(df)
    sales_and_rating_over_time(df)
    scatter_sales_vs_rating(df)
    correlation_heatmap(df)
    boxplot_gross_income(df)

    ans = advanced_answers(df)
    with open(OUT_RESULTS / "advanced_answers.txt", "w", encoding="utf-8") as f:
        for k, v in ans.items():
            f.write(f"{k}: {v}\n")

    print("Done. Check results/")


if __name__ == "__main__":
    main()
