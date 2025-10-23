import streamlit as st
import os
import time
from threading import Thread, Event
import queue
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sumo_simulator import SUMOSimulator
import dashscope
from dashscope import Generation
import json
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots

class TrafficAnalyzer:
    def __init__(self, api_key: str = 'sk-bc4339fe8e8e48f6b57fab86b5d70afa'):
        self.api_key = api_key
        dashscope.api_key = api_key
        self.junction_desc = "该路口是成都市的倪家桥路与领事馆路交叉口。有4个相位，分别是：1.南北方向直行与右转（南北方向各有5条车道（4直行+1右转））；2.南北方向左转（南北方向各有1条左转车道）；3.东向西通行（东方向2条车道）；4.西向东通行（西方向2条车道）"

    def analyze_traffic(self, vehicle_counts, phase_info):
        prompt = f"""作为交通信号配时专家，{self.junction_desc}请根据以下模板简要分析交通状况：

当前交通状况：
{self._format_counts(vehicle_counts)}

当前相位：{phase_info['phase_name']}
剩余时间：{phase_info['remaining_duration']:.1f}秒

请按以下格式回答：
1. 当前状态：[拥堵/正常/空闲]
2. 建议操作：[保持当前相位/切换到XX相位/延长当前相位XX秒]

只需给出简短的状态判断和一条具体建议。"""

        try:
            response = Generation.call(
                model='qwen-plus',
                prompt=prompt,
                seed=1234,
                max_tokens=200,
                temperature=0.1,
                top_p=0.8,
                result_format='message',
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                return f"API调用失败: {response.code}"
                
        except Exception as e:
            return f"分析失败: {str(e)}"

    def _format_counts(self, counts):
        return "\n".join([f"{direction}向: {count}辆" for direction, count in counts.items()])

class SimulationManager:
    def __init__(self):
        self.simulator = None
        self.sim_thread = None
        self.stop_event = Event()
        self.signalized_junctions = []
        
    def start_simulation(self, config_file = os.path.join(os.getcwd(), "sumo_llm/osm.sumocfg"), junctions_file = os.path.join(os.getcwd(), "sumo_llm/J54_data.json")):
        
        if self.simulator is None:
            self.simulator = SUMOSimulator(config_file, junctions_file, gui=True)
            if self.simulator.start_simulation():
                # 获取所有带信号灯的路口
                self.signalized_junctions = self._get_signalized_junctions()
                # 启动仿真线程
                self.stop_event.clear()
                self.sim_thread = Thread(target=self._run_simulation)
                self.sim_thread.daemon = True
                self.sim_thread.start()
                return True
        return False
    
    def _get_signalized_junctions(self):
        signalized = []
        for junction_id, data in self.simulator.junctions_data.items():
            if data.get("type") == "traffic_light":  # 检查type字段是否为traffic_light
                signalized.append({
                    'id': junction_id,
                    'name': data.get('junction_name', junction_id)
                })
        return signalized[:3]
    
    def _run_simulation(self):
        while not self.stop_event.is_set():
            if not self.simulator.step():
                break
            time.sleep(1)
    
    def get_junction_state(self, junction_id):
        if self.simulator and self.simulator.is_connected():
            try:
                counts, _ = self.simulator.get_junction_vehicle_counts(junction_id)
                phase_info = self.simulator.get_current_phase(junction_id)
                print(counts, phase_info)
                return counts, phase_info
            except Exception as e:
                st.error(f"获取路口状态失败: {str(e)}")
        return None, None
    
    def stop_simulation(self):
        self.stop_event.set()
        if self.simulator:
            self.simulator.close()
        self.simulator = None

def create_traffic_flow_plots(counts):
    if not counts:
        return None
    
    try:
        # 准备数据
        directions = list(counts.keys())
        vehicle_counts = [counts[direction]['vehicle_count'] for direction in directions]
        mean_speeds = [counts[direction]['mean_speed'] for direction in directions]
        halting_counts = [counts[direction]['halting_count'] for direction in directions]
        waiting_times = [counts[direction]['waiting_time'] for direction in directions]
        
        # 缩短方向名称以便更好显示
        short_directions = [d[:10] + '...' if len(d) > 10 else d for d in directions]
        
        # 创建2x2的子图布局
        fig = plotly.subplots.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>车辆数量</b>', 
                '<b>平均速度 (m/s)</b>', 
                '<b>停车数量</b>', 
                '<b>等待时间 (s)</b>'
            ),
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )
        
        # 添加车辆数量柱状图
        fig.add_trace(
            go.Bar(
                x=short_directions,
                y=vehicle_counts,
                text=vehicle_counts,
                textposition='auto',
                marker_color='#1f77b4'
            ),
            row=1, col=1
        )
        
        # 添加平均速度柱状图
        fig.add_trace(
            go.Bar(
                x=short_directions,
                y=mean_speeds,
                text=[f'{speed:.1f}' for speed in mean_speeds],
                textposition='auto',
                marker_color='#ff7f0e'
            ),
            row=1, col=2
        )
        
        # 添加停车数量柱状图
        fig.add_trace(
            go.Bar(
                x=short_directions,
                y=halting_counts,
                text=halting_counts,
                textposition='auto',
                marker_color='#2ca02c'
            ),
            row=2, col=1
        )
        
        # 添加等待时间柱状图
        fig.add_trace(
            go.Bar(
                x=short_directions,
                y=waiting_times,
                text=[f'{time:.1f}' for time in waiting_times],
                textposition='auto',
                marker_color='#d62728'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            height=800,
            width=1000,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        )
        
        # 更新所有子图的x轴和y轴样式
        fig.update_xaxes(
            tickangle=30,
            showgrid=False,
            tickfont=dict(size=9)
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128,128,128,0.2)',
        )
        
        # 为每个子图添加标题样式
        for annotation in fig.layout.annotations:
            annotation.update(font=dict(size=12, color='black', family='Arial'))
        
        return fig
    
    except Exception as e:
        st.error(f"创建图表时出错: {str(e)}")
        st.write("完整错误信息:", e)
        return None

def initialize_session_state(qwen_key):
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'sim_manager' not in st.session_state:
        st.session_state.sim_manager = SimulationManager()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = TrafficAnalyzer()
    if 'selected_junction' not in st.session_state:
        st.session_state.selected_junction = None
    if 'current_counts' not in st.session_state:
        st.session_state.current_counts = None

def main():
    st.title("SUMO交通仿真分析系统")
    
    qwen_key = "sk-"
    initialize_session_state(qwen_key)
    
    # 侧边栏控制
    with st.sidebar:
        st.header("系统控制")
        
        # 仿真控制
        st.subheader("仿真控制")
        if not st.session_state.simulation_running:
            if st.button("启动仿真"):
                current_dir = os.getcwd()
                config_file = os.path.join(current_dir, "sumo_llm", "osm.sumocfg")
                junctions_file = os.path.join(current_dir, "sumo_llm", "J54_data.json")
                
                if st.session_state.sim_manager.start_simulation(config_file, junctions_file):
                    st.session_state.simulation_running = True
                    st.success("仿真启动成功！")
                else:
                    st.error("仿真启动失败！")
        else:
            if st.button("停止仿真"):
                st.session_state.sim_manager.stop_simulation()
                st.session_state.simulation_running = False
                st.warning("仿真已停止")
        
        if st.session_state.simulation_running:
            st.subheader("功能选择")
            analysis_mode = st.radio(
                "选择分析模式",
                ["AI分析", "交通状态可视化"]
            )
            
            junctions = st.session_state.sim_manager.signalized_junctions
            if junctions:
                selected_junction = st.selectbox(
                    "选择路口",
                    options=junctions,
                    format_func=lambda x: x['name']
                )
                
                if st.button("开始分析"):
                    st.session_state.selected_junction = selected_junction
                    st.session_state.analysis_mode = analysis_mode
                    counts, phase_info = st.session_state.sim_manager.get_junction_state(selected_junction['id'])
                    st.session_state.current_counts = counts
                    st.session_state.current_phase_info = phase_info
            else:
                st.warning("未找到带信号灯的路口")
    
    # 主界面
    if st.session_state.simulation_running:
        if hasattr(st.session_state, 'analysis_mode') and hasattr(st.session_state, 'selected_junction'):
            if st.session_state.analysis_mode == "AI分析":
                st.header(f"AI分析结果 - {st.session_state.selected_junction['name']}")
                
                if hasattr(st.session_state, 'current_counts') and hasattr(st.session_state, 'current_phase_info'):
                    counts = st.session_state.current_counts
                    phase_info = st.session_state.current_phase_info
                    
                    if counts and phase_info:
                        # 显示当前状态
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("当前车流量")
                            for direction, count in counts.items():
                                st.text(f"{direction}向: {count}辆")
                        
                        with col2:
                            st.subheader("当前信号相位")
                            st.text(f"相位名称: {phase_info['phase_name']}")
                            st.text(f"剩余时间: {phase_info['remaining_duration']:.1f}秒")
                        
                        # LLM分析
                        st.subheader("AI分析建议")
                        with st.spinner("正在分析..."):
                            analysis = st.session_state.analyzer.analyze_traffic(counts, phase_info)
                            st.info(analysis)
                    else:
                        st.error("无法获取路口数据")
                        
            elif st.session_state.analysis_mode == "交通状态可视化":
                st.header(f"交通状态可视化 - {st.session_state.selected_junction['name']}")
                
                if hasattr(st.session_state, 'current_counts'):
                    counts = st.session_state.current_counts
                    if counts:
                        fig = create_traffic_flow_plots(counts)  # 注意函数名变更
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("无法获取路口数据")
    else:
        st.info("请先启动仿真")
    return st

simulation_manager = SimulationManager()
simulation_manager.start_simulation()

if __name__ == "__main__":
    main()