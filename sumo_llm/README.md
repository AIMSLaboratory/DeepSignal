# SUMO Traffic Simulation Analysis System

一个基于SUMO（Simulation of Urban MObility）的交通仿真分析系统，集成了AI分析和实时可视化功能，帮助交通工程师更好地理解和优化交通流量。

## 功能特点

- **实时交通仿真**：基于SUMO进行微观交通仿真，支持复杂路网模拟
- **AI辅助分析**：集成Qwen大模型，提供专业的交通状态分析和优化建议
- **实时数据可视化**：使用Plotly绘制直观的交通流量图表
- **交互式界面**：基于Streamlit构建的用户友好界面
- **多路口监控**：支持同时监控多个信号交叉口

## 界面展示

### AI分析界面
![AI Analysis Interface](images/LLM分析界面.png)
*展示实时交通状态和AI分析结果*

### 可视化界面
![Visualization Interface](images/统计图和仿真界面.png)
*统计图表与某路口的仿真*

## 技术栈

- SUMO：交通仿真引擎
- Python：主要开发语言
- Streamlit：Web界面框架
- Plotly：数据可视化
- Qwen：AI分析模型

## 环境配置

### SUMO安装
1. 从[SUMO官网](https://sumo.dlr.de/docs/Downloads.php)下载并安装SUMO
2. 设置环境变量：
   - Windows: 将SUMO安装目录添加到PATH
   - Linux/Mac: export SUMO_HOME="/usr/local/share/sumo"

### Python依赖
```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行程序

```
找到“st.session_state.analyzer = TrafficAnalyzer(api_key="sk-key")”替换成你的qwen api-key

streamlit run app.py
```

## 未来改进计划

1. **多模型集成**
   - 添加function calling功能
   - 支持自定义提示词模板
   - 添加多智能体模型框架，根据交通状况调用不同的控制算法
2. **高级可视化功能**
   - 添加热力图展示交通拥堵状况
   - 支持历史数据回放功能
   - 增加3D路网可视化