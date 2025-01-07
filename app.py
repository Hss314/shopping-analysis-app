import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# 設定 Seaborn 主題樣式
sns.set_theme(style="whitegrid", palette="muted")

# 讀取資料集
df = pd.read_csv("shopping_data.csv")

# 顯示標題
st.title("線上購物資料分析")

# 資料集展示
st.subheader("資料集")
st.write(df.head())

# 顧客購物模式分析 - K-means 分群
def kmeans_analysis():
    # 資料清理：選擇需要的欄位，並移除缺失值
    columns_to_use = ["Tenure_Months", "Quantity", "Online_Spend", "Offline_Spend"]
    df_clean = df[columns_to_use].dropna()

    # 標準化資料以便進行 K-means 分群
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_clean)

    # 使用肘部法則 (Elbow Method) 確定最佳的分群數量
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_scaled)
        inertia.append(kmeans.inertia_)

    # 顯示肘部法則的視覺化結果
    st.subheader("Elbow Method for Optimal K")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertia, marker='o', linestyle='-', color='#2E8B57', markersize=8, linewidth=2)
    ax.set_title("Elbow Method for Optimal K", fontsize=16, fontweight='bold')
    ax.set_xlabel("Number of Clusters (K)", fontsize=14)
    ax.set_ylabel("Inertia", fontsize=14)
    st.pyplot(fig)

    # 假設最佳分群數為 3（根據肘部法則）
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(data_scaled)

    # 查看各分群的平均特徵值
    cluster_summary = df_clean.groupby("Cluster").mean()
    st.subheader("各分群的平均特徵值：")
    st.write(cluster_summary)

    # 可視化分群結果
    st.subheader("顧客分群結果")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df_clean["Tenure_Months"], y=df_clean["Online_Spend"], hue=df_clean["Cluster"], palette="Set2", s=30, alpha=0.5, edgecolor='black', linewidth=0.2, ax=ax)
    ax.set_title("Customer Clusters based on Shopping Behavior", fontsize=16, fontweight='bold')
    ax.set_xlabel("Tenure (Months)", fontsize=14)
    ax.set_ylabel("Online Spend", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5)
    st.pyplot(fig)

# 促銷和折扣影響分析
def discount_analysis():
    # 資料清理：移除缺失值並確認資料型別
    df_clean = df.dropna(subset=["Coupon_Status", "Discount_pct", "Online_Spend"])
    df_clean["Coupon_Status"] = df_clean["Coupon_Status"].astype("category")
    df_clean["Discount_pct"] = df_clean["Discount_pct"].astype(float)
    df_clean["Online_Spend"] = df_clean["Online_Spend"].astype(float)

    # 顯示有無使用優惠券的分佈
    st.subheader("Distribution of Coupon Usage")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="Coupon_Status", data=df_clean, palette="Set2", ax=ax)
    ax.set_title("Distribution of Coupon Usage", fontsize=16, fontweight="bold")
    ax.set_xlabel("Coupon Status", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    st.pyplot(fig)

    # 比較有無使用優惠券的消費平均值
    coupon_yes = df_clean[df_clean["Coupon_Status"] == "Used"]["Online_Spend"]
    coupon_no = df_clean[df_clean["Coupon_Status"] == "Not Used"]["Online_Spend"]

    st.write("平均消費額 (使用優惠券):", coupon_yes.mean())
    st.write("平均消費額 (未使用優惠券):", coupon_no.mean())

    # 進行 T 檢定
    t_stat, p_value = ttest_ind(coupon_yes, coupon_no, equal_var=False)
    st.write("T 檢定統計量:", t_stat)
    st.write("P 值:", p_value)

    if p_value < 0.05:
        st.write("結果：優惠券顯著影響消費金額 (p < 0.05)")
    else:
        st.write("結果：優惠券對消費金額無顯著影響 (p >= 0.05)")

    # 使用 Violin Plot 查看分佈
    st.subheader("Violin Plot of Online Spend by Coupon Status")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x="Coupon_Status", y="Online_Spend", data=df_clean, palette="muted", inner="quartile", ax=ax)
    ax.set_title("Online Spend by Coupon Status (Violin Plot)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Coupon Status", fontsize=14)
    ax.set_ylabel("Online Spend", fontsize=14)
    st.pyplot(fig)

    # 使用 Hexbin Plot 展示數據密集度
    st.subheader("Hexbin Plot of Discount Percentage vs Online Spend")
    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(df_clean["Discount_pct"], df_clean["Online_Spend"], gridsize=30, cmap="Blues", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    ax.set_title("Discount Percentage vs Online Spend (Hexbin Plot)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Discount Percentage (%)", fontsize=14)
    ax.set_ylabel("Online Spend", fontsize=14)
    st.pyplot(fig)

# 產品銷售和趨勢分析
def sales_trends_analysis():
    # 清理資料：移除缺失值並處理資料型別
    df_clean = df.dropna(subset=["Quantity", "Avg_Price", "Product_Category", "Transaction_Date"])

    # 轉換日期型別
    df_clean["Transaction_Date"] = pd.to_datetime(df_clean["Transaction_Date"])

    # 計算每筆交易的銷售額
    df_clean["Sales_Amount"] = df_clean["Quantity"] * df_clean["Avg_Price"]

    # 提取月份和年分
    df_clean["Year_Month"] = df_clean["Transaction_Date"].dt.to_period("M").astype(str)

    # 按月份計算總銷售額
    monthly_sales = df_clean.groupby("Year_Month")["Sales_Amount"].sum().reset_index()

    st.subheader("Monthly Sales Trend")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=monthly_sales, x="Year_Month", y="Sales_Amount", marker='o', color='b', ax=ax)
    ax.set_title("Monthly Sales Trend", fontsize=16)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Sales Amount", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(True)
    st.pyplot(fig)

    # 按產品類別分析銷售額
    category_sales = df_clean.groupby("Product_Category")["Sales_Amount"].sum().reset_index()

    # 只顯示銷售額前 10 的產品類別
    top_categories = category_sales.sort_values("Sales_Amount", ascending=False).head(10)

    st.subheader("Sales Amount by Top 10 Product Categories")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_categories, x="Sales_Amount", y="Product_Category", palette="viridis", ax=ax)
    ax.set_title("Sales Amount by Top 10 Product Categories", fontsize=16)
    ax.set_xlabel("Total Sales Amount", fontsize=12)
    ax.set_ylabel("Product Category", fontsize=12)
    st.pyplot(fig)

    # 按產品類別分析銷售額
    category_sales = df_clean.groupby("Product_Category")["Sales_Amount"].sum().reset_index()

    # 只顯示銷售額前 10 的產品類別
    top_categories = category_sales.sort_values("Sales_Amount", ascending=False).head(10)

    st.subheader("Monthly Sales Trend by Top Product Categories")
    df_top_categories = df_clean[df_clean["Product_Category"].isin(top_categories["Product_Category"])]

    # 確保 'Year_Month' 按日期順序排序
    df_top_categories["Year_Month"] = pd.to_datetime(df_top_categories["Year_Month"], format='%Y-%m')
    df_top_categories = df_top_categories.sort_values("Year_Month")

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.lineplot(data=df_top_categories, x="Year_Month", y="Sales_Amount", hue="Product_Category", marker='o', palette="Set2", ax=ax)

    # 設定標題與軸標籤
    ax.set_title("Monthly Sales Trend by Top Product Categories", fontsize=16)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Total Sales Amount", fontsize=12)

    # 設定 x 軸的每月標籤
    all_dates = pd.date_range(df_top_categories["Year_Month"].min(), df_top_categories["Year_Month"].max(), freq='MS')  # 每月起始
    ax.set_xticks(all_dates)  # 設定所有月份為 x 軸刻度
    ax.set_xticklabels([date.strftime('%Y-%m') for date in all_dates], rotation=45)  # 設定格式並旋轉

    ax.grid(True)
    ax.legend(title="Product Category", loc="upper left", bbox_to_anchor=(1, 1))
    st.pyplot(fig)

# 使用選擇框讓用戶選擇分析
analysis_options = ["顧客購物模式分析", "促銷和折扣影響分析", "產品銷售和趨勢分析"]
selection = st.sidebar.selectbox("選擇分析", analysis_options)

if selection == "顧客購物模式分析":
    kmeans_analysis()
elif selection == "促銷和折扣影響分析":
    discount_analysis()
elif selection == "產品銷售和趨勢分析":
    sales_trends_analysis()
