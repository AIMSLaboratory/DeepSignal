import streamlit as st
import os
import time
from threading import Thread, Event
from sumo_simulator import SUMOSimulator
import dashscope
from dashscope import Generation


class TrafficAnalyzer:
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

class SimulationManager:
    def __init__(self):
        self.simulator = None
        self.sim_thread = None
        self.stop_event = Event()
        self.signalized_junctions = []
        
    def start_simulation(self, config_file, junctions_file):
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
                return counts, phase_info
            except Exception as e:
                st.error(f"获取路口状态失败: {str(e)}")
        return None, None
    
    def stop_simulation(self):
        self.stop_event.set()
        if self.simulator:
            self.simulator.close()
        self.simulator = None

def initialize_session_state():
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'sim_manager' not in st.session_state:
        st.session_state.sim_manager = SimulationManager()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = TrafficAnalyzer(api_key="sk-key")

def main():
    st.title("SUMO交通仿真分析系统")
    
    # 初始化session state
    initialize_session_state()
    
    # 侧边栏：启动/停止仿真
    with st.sidebar:
        st.header("仿真控制")
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
    
    # 主界面
    if st.session_state.simulation_running:
        # 选择要分析的路口
        junctions = st.session_state.sim_manager.signalized_junctions
        if junctions:
            selected_junction = st.selectbox(
                "选择要分析的路口",
                options=junctions,
                format_func=lambda x: x['name']
            )
            
            # 分析按钮
            if st.button("分析当前路口"):
                counts, phase_info = st.session_state.sim_manager.get_junction_state(selected_junction['id'])
                
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
                    st.subheader("AI分析结果")
                    with st.spinner("正在分析..."):
                        analysis = st.session_state.analyzer.analyze_traffic(counts, phase_info)
                        st.info(analysis)
                else:
                    st.error("无法获取路口数据")
        else:
            st.warning("未找到带信号灯的路口")
    else:
        st.info("请先启动仿真")

if __name__ == "__main__":
    main()