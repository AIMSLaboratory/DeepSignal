import streamlit as st
import os
import time
from threading import Thread, Event
from sumo_simulator import SUMOSimulator
import dashscope
from dashscope import Generation
import plotly.graph_objects as go
from collections import deque
import pandas as pd
from typing import Dict, List

class TrafficAnalyzer:
    """处理交通数据AI分析的类"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        dashscope.api_key = api_key

    def analyze_traffic(self, vehicle_counts, phase_info):
        prompt = f"""作为交通信号配时专家，请根据以下模板简要分析交通状况：

当前车流量：
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

class TrafficDataCollector:
    """收集和管理交通数据的类"""
    def __init__(self, max_history_minutes=20):
        self.max_points = max_history_minutes
        self.data = {}  # 格式: {junction_id: {direction: deque([])}}
        self.collection_interval = 60  # 60步
        self.step_count = 0  # 添加步数计数器
    
    def initialize_junction(self, junction_id: str, directions: List[str], step: int):
        """初始化路口数据"""
        if junction_id not in self.data:
            initial_steps = [step - i * self.collection_interval 
                           for i in range(self.max_points)][::-1]
            self.data[junction_id] = {
                direction: deque([(t, 0) for t in initial_steps], 
                                maxlen=self.max_points)
                for direction in directions
            }
    
    def update_data(self, junction_id: str, new_counts: Dict[str, int]):
        """更新路口数据"""
        self.step_count += 1
        
        if junction_id not in self.data:
            self.initialize_junction(junction_id, new_counts.keys(), self.step_count)
        
        for direction, count in new_counts.items():
            if direction in self.data[junction_id]:
                last_step = self.data[junction_id][direction][-1][0]
                if self.step_count - last_step >= self.collection_interval:
                    self.data[junction_id][direction].append((self.step_count, count))
    
    def get_junction_data(self, junction_id: str) -> Dict[str, pd.DataFrame]:
        """获取路口数据"""
        if junction_id not in self.data:
            return {}
        
        result = {}
        for direction, data in self.data[junction_id].items():
            steps, counts = zip(*list(data))
            formatted_times = [self.format_step_time(t) for t in steps]
            df = pd.DataFrame({
                'time': formatted_times,
                'step': steps,
                'count': counts
            })
            result[direction] = df
        
        return result
    
    @staticmethod
    def format_step_time(step: int) -> str:
        """将步数转换为可读格式 (MM:SS)"""
        # 假设每步是1秒
        minutes = step // 60
        seconds = step % 60
        return f"{minutes:02d}:{seconds:02d}"

class SimulationManager:
    """管理仿真和数据收集的类"""
    def __init__(self):
        self.simulator = None
        self.sim_thread = None
        self.data_collection_thread = None
        self.stop_event = Event()
        self.signalized_junctions = []
        self.data_collector = TrafficDataCollector()
    
    def start_simulation(self, config_file, junctions_file):
        """启动仿真"""
        if self.simulator is None:
            self.simulator = SUMOSimulator(config_file, junctions_file, gui=True)
            if self.simulator.start_simulation():
                self.signalized_junctions = self._get_signalized_junctions()
                print(self.signalized_junctions)
                self.stop_event.clear()
                
                # 启动仿真线程
                self.sim_thread = Thread(target=self._run_simulation)
                self.sim_thread.daemon = True
                self.sim_thread.start()
                
                # 启动数据收集线程
                self.data_collection_thread = Thread(target=self._collect_traffic_data)
                self.data_collection_thread.daemon = True
                self.data_collection_thread.start()
                return True
        return False
    
    def _get_signalized_junctions(self):
        """获取信号灯路口"""
        signalized = []
        for junction_id, data in self.simulator.junctions_data.items():
            if data.get("type") == "traffic_light":
                signalized.append({
                    'id': junction_id,
                    'name': data.get('junction_name', junction_id)
                })
        return signalized[-2:]
    
    def _run_simulation(self):
        """运行仿真的线程函数"""
        while not self.stop_event.is_set():
            if not self.simulator.step():
                break
            time.sleep(0.1)
    
    def _collect_traffic_data(self):
        """收集交通数据的线程函数"""
        while not self.stop_event.is_set():
            if self.simulator and self.simulator.is_connected():
                for junction in self.signalized_junctions:
                    try:
                        counts, _ = self.simulator.get_junction_vehicle_counts(junction['id'])
                        if counts:
                            self.data_collector.update_data(junction['id'], counts)
                    except Exception as e:
                        print(f"数据收集错误: {str(e)}")
                time.sleep(1)
    
    def get_junction_state(self, junction_id):
        """获取路口状态"""
        if self.simulator and self.simulator.is_connected():
            try:
                counts, _ = self.simulator.get_junction_vehicle_counts(junction_id)
                phase_info = self.simulator.get_current_phase(junction_id)
                return counts, phase_info
            except Exception as e:
                st.error(f"获取路口状态失败: {str(e)}")
        return None, None
    
    def stop_simulation(self):
        """停止仿真"""
        self.stop_event.set()
        if self.simulator:
            self.simulator.close()
        self.simulator = None

def create_traffic_plot(data: Dict[str, pd.DataFrame], title: str):
    """创建交通流量趋势图"""
    fig = go.Figure()
    
    for direction, df in data.items():
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['count'],
            name=f"{direction}向",
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="仿真时间 (MM:SS)",
        yaxis_title="车辆数",
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def initialize_session_state():
    """初始化Streamlit会话状态"""
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'sim_manager' not in st.session_state:
        st.session_state.sim_manager = SimulationManager()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = TrafficAnalyzer(api_key="sk-key")
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "AI分析"

def main():
    """主函数"""
    st.title("SUMO交通仿真分析系统")
    
    initialize_session_state()
    
    # 侧边栏
    with st.sidebar:
        st.header("系统控制")
        
        # 仿真控制
        st.subheader("仿真控制")
        if not st.session_state.simulation_running:
            if st.button("启动仿真"):
                current_dir = os.getcwd()
                config_file = os.path.join(current_dir, "osm.sumocfg")
                junctions_file = os.path.join(current_dir, "junctions_data.json")
                
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
        
        # 功能选择
        st.subheader("功能选择")
        st.radio(
            "选择功能",
            ["AI分析", "交通流量可视化"],
            key="selected_tab"
        )
    
    # 主界面
    if st.session_state.simulation_running:
        junctions = st.session_state.sim_manager.signalized_junctions
        if junctions:
            selected_junction = st.selectbox(
                "选择要分析的路口",
                options=junctions,
                format_func=lambda x: x['name']
            )
            
            if st.session_state.selected_tab == "AI分析":
                # AI分析部分
                counts, phase_info = st.session_state.sim_manager.get_junction_state(selected_junction['id'])
                
                if counts and phase_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("当前车流量")
                        for direction, count in counts.items():
                            st.text(f"{direction}向: {count}辆")
                    
                    with col2:
                        st.subheader("当前信号相位")
                        st.text(f"相位名称: {phase_info['phase_name']}")
                        st.text(f"剩余时间: {phase_info['remaining_duration']:.1f}秒")
                    
                    if st.button("进行AI分析"):
                        st.subheader("AI分析结果")
                        with st.spinner("正在分析..."):
                            analysis = st.session_state.analyzer.analyze_traffic(counts, phase_info)
                            st.info(analysis)
                else:
                    st.error("无法获取路口数据")
            
            else:  # 交通流量可视化
                st.subheader("交通流量趋势")
                junction_data = st.session_state.sim_manager.data_collector.get_junction_data(selected_junction['id'])
                
                if junction_data:
                    fig = create_traffic_plot(
                        junction_data,
                        f"路口 {selected_junction['name']} 交通流量趋势"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 自动刷新
                time.sleep(5)
                st.rerun()
        
        else:
            st.warning("未找到带信号灯的路口")
    else:
        st.info("请先启动仿真")

if __name__ == "__main__":
    main()