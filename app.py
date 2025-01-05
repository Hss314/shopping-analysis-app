import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取資料
@st.cache
def load_data():
    data = pd.read_csv("shopping_data.csv")
    return data

data = load_data()

# 應用標題
st.title("線上購物行為分析儀表板")

# 篩選條件
st.sidebar.header("篩選條件")
selected_month = st.sidebar.selectbox("選擇月份", data['Month'].unique())
selected_category = st.sidebar.multiselect("選擇產品類別", data['Product_Category'].unique())

# 篩選資料
filtered_data = data[(data['Month'] == selected_month)]
if selected_category:
    filtered_data = filtered_data[filtered_data['Product_Category'].isin(selected_category)]

# 顯示數據表
st.write(f"篩選後的資料：{filtered_data.shape[0]} 筆")
st.dataframe(filtered_data)

# 銷售趨勢圖
st.subheader("銷售趨勢分析")
fig, ax = plt.subplots()
sns.lineplot(data=filtered_data, x='Transaction_Date', y='Quantity', hue='Product_Category', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# 優惠券使用對消費的影響
st.subheader("優惠券使用對消費的影響")
coupon_usage = filtered_data.groupby('Coupon_Status')['Online_Spend'].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=coupon_usage, x='Coupon_Status', y='Online_Spend', ax=ax)
st.pyplot(fig)
