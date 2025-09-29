import streamlit as st
import pandas as pd
import numpy as np

# 页面设置
st.set_page_config(
    page_title="Streamlit 示例",
    page_icon="📊",
    layout="wide"
)

# 标题
st.title("Streamlit 示例应用")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("控制面板")
    user_name = st.text_input("请输入你的名字")
    slider_value = st.slider("选择一个数值", 0, 100, 50)

# 主界面
col1, col2 = st.columns(2)

with col1:
    st.subheader("基本功能演示")
    st.write(f"你好，{user_name}！你选择的数值是：{slider_value}")

    # 按钮
    if st.button("点击惊喜"):
        st.balloons()

    # 复选框
    if st.checkbox("显示数据示例"):
        data = pd.DataFrame({
            'A': np.random.randn(10),
            'B': np.random.rand(10) * 100
        })
        st.dataframe(data.style.highlight_max(axis=0))

with col2:
    st.subheader("图表演示")

    # 生成示例数据
    chart_data = pd.DataFrame(
        np.random.randn(slider_value, 3),
        columns=['a', 'b', 'c']
    )

    # 折线图
    st.line_chart(chart_data)

    # 地图示例
    if st.checkbox("显示地图"):
        map_data = pd.DataFrame(
            np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
            columns=['lat', 'lon']
        )
        st.map(map_data)

# 文件上传示例
st.markdown("---")
st.subheader("文件上传功能")
uploaded_file = st.file_uploader("上传CSV文件", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("前5行数据：")
    st.table(df.head())

# 进度条演示
st.markdown("---")
st.subheader("进度条演示")
import time

if st.button("开始处理"):
    progress_bar = st.progress(0)
    status_text = st.empty()

    for percent_complete in range(100):
        time.sleep(0.05)
        progress_bar.progress(percent_complete + 1)
        status_text.text(f"处理进度：{percent_complete + 1}%")

    status_text.text("处理完成！")
    st.success("✅ 任务已完成")