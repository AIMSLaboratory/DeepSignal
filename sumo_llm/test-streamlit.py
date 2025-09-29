import streamlit as st
import pandas as pd
import numpy as np

# 标题
st.title("数据分析演示")

# 上传文件
uploaded_file = st.file_uploader("上传 CSV 文件")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("数据预览:", data.head())

# 滑动条控制图表
num_points = st.slider("选择数据点数量", 10, 100)
x = np.random.randn(num_points)
st.line_chart(x)

# 按钮触发操作
if st.button("生成报告"):
    st.success("分析完成！平均值为: {:.2f}".format(np.mean(x)))