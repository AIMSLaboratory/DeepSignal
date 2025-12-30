import os
import sys
import traci
import json
import time
import datetime
from collections import defaultdict
from threading import Thread, Event
from typing import Optional, Dict
import csv

# 确保SUMO_HOME环境变量已设置
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("请设置SUMO_HOME环境变量")

_UNSET = object()

class SUMOSimulator:
    def __init__(self, config_file=os.path.join(os.getcwd(), "sumo_llm/osm.sumocfg"), junctions_file=os.path.join(os.getcwd(), "sumo_llm/J54_data.json"), gui=True, history_file=None):
        """
        初始化SUMO仿真器
        config_file: SUMO配置文件路径 (.sumocfg)
        junctions_file: 路口数据JSON文件路径
        gui: 是否使用GUI模式
        history_file: 历史数据存储文件路径
        """
        self.config_file = os.path.abspath(config_file)
        self.junctions_file = os.path.abspath(junctions_file) if junctions_file else None
        self.gui = gui
        self.simulation_started = False
        self.warmup_done = False
        self.warmup_steps = 300
        self.start_time = None
        self.vehicle_counts = defaultdict(lambda: defaultdict(list))
        self.timestamps = defaultdict(lambda: defaultdict(list))
        
        # 历史数据存储
        self.history_file = history_file or os.path.join(os.path.dirname(config_file), "traffic_history.json")
        self.historical_data = {
            'timestamps': [],
            'phase_queues': [],
            'phases': []
        }
        self.load_historical_data()
        
        # 验证文件存在
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"SUMO配置文件未找到: {self.config_file}")
        if self.junctions_file and (not os.path.exists(self.junctions_file)):
            raise FileNotFoundError(f"路口数据文件未找到: {self.junctions_file}")
        
        # 加载路口配置
        if self.junctions_file:
            try:
                with open(self.junctions_file, 'r', encoding='utf-8') as f:
                    self.junctions_data = json.load(f)
            except Exception as e:
                print(f"Error loading junctions data: {str(e)}")
                self.junctions_data = {}
        else:
            self.junctions_data = {}

    def load_historical_data(self):
        """加载历史数据"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.historical_data = json.load(f)
                # 确保数据结构正确
                if not all(key in self.historical_data for key in ['timestamps', 'phase_queues', 'phases']):
                    self.historical_data = {
                        'timestamps': [],
                        'phase_queues': [],
                        'phases': []
                    }
        except Exception as e:
            print(f"加载历史数据失败: {str(e)}")
            self.historical_data = {
                'timestamps': [],
                'phase_queues': [],
                'phases': []
            }

    def save_historical_data(self):
        """保存历史数据"""
        try:
            # 只保留最近24小时的数据
            max_history = 24 * 3600  # 24小时的秒数
            current_time = datetime.datetime.now()
            
            while len(self.historical_data['timestamps']) > 0:
                timestamp = datetime.datetime.fromisoformat(self.historical_data['timestamps'][0])
                if (current_time - timestamp).total_seconds() > max_history:
                    for key in self.historical_data:
                        self.historical_data[key].pop(0)
                else:
                    break
            
            # 保存到文件
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.historical_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史数据失败: {str(e)}")

    def collect_traffic_data(self, tl_id: str):
        """收集交通数据并保存到历史记录"""
        try:
            # 获取当前时间
            current_time = datetime.datetime.now()
            
            # 获取相位队列数据
            phase_queues = self.calculate_all_phases_pressure(tl_id)
            
            # 获取当前相位信息
            phase_info = self.get_current_phase(tl_id)
            current_phase = phase_info.get('phase_index', 0)
            
            # 保存数据
            self.historical_data['timestamps'].append(current_time.isoformat())
            self.historical_data['phase_queues'].append(phase_queues)
            self.historical_data['phases'].append(current_phase)
            
            # 定期保存到文件
            if len(self.historical_data['timestamps']) % 10 == 0:  # 每10条数据保存一次
                self.save_historical_data()
                
        except Exception as e:
            print(f"收集交通数据失败: {str(e)}")

    def get_historical_data(self, tl_id: str, time_window: Optional[int] = None):
        """获取历史数据
        
        Args:
            tl_id: 交通信号灯ID
            time_window: 时间窗口（秒），None表示使用全部历史数据
            
        Returns:
            历史数据字典
        """
        if time_window is None:
            return self.historical_data
        
        # 按仿真时间步数取最近 time_window 条记录（step-length=1.0 时等价于秒）
        # 这里不再使用墙上时间，避免不同模型推理速度导致窗口长度不一致
        total = len(self.historical_data.get('timestamps', []))
        if total == 0:
            return {
                'timestamps': [],
                'phase_queues': [],
                'phases': []
            }
        start_idx = max(total - time_window, 0)
        return {
            'timestamps': self.historical_data['timestamps'][start_idx:],
            'phase_queues': self.historical_data['phase_queues'][start_idx:],
            'phases': self.historical_data['phases'][start_idx:]
        }

    def is_connected(self):
        """检查是否连接到SUMO"""
        try:
            return traci.getConnection() is not None
        except:
            return False

    def start_simulation(self):
        """启动仿真并进行预热"""
        try:
            if not self.simulation_started:
                # 如果已经有连接，先关闭
                try:
                    if self.is_connected():
                        traci.close()
                except:
                    pass
                
                # 设置SUMO命令
                sumo_home = os.environ.get('SUMO_HOME', '')
                if sumo_home:
                    sumo_binary = os.path.join(sumo_home, "sumo-gui" if self.gui else "sumo")
                else:
                    sumo_binary = "sumo-gui" if self.gui else "sumo"
                sumo_cmd = [
                    sumo_binary,
                    "-c", self.config_file,
                    "--step-length", "1.0",
                    "--no-warnings", "true",
                    "--start",
                    "--quit-on-end",
                ]
                
                print(f"Starting SUMO with command: {' '.join(sumo_cmd)}")
                
                # 启动SUMO
                traci.start(sumo_cmd)
                self.simulation_started = True
                print("Successfully connected to SUMO")
                
                print("Starting warmup phase...")
                # 预热阶段
                for i in range(self.warmup_steps):
                    if not self.is_connected():
                        raise Exception("SUMO connection lost during warmup")
                    traci.simulationStep()
                    if i % 100 == 0:
                        print(f"Warmup progress: {i}/{self.warmup_steps}")
                
                self.warmup_done = True
                self.start_time = time.time()
                print("Warmup completed. Starting real-time simulation.")
                return True
                
        except Exception as e:
            print(f"Error starting simulation: {str(e)}")
            self.simulation_started = False
            self.warmup_done = False
            try:
                if self.is_connected():
                    traci.close()
            except:
                pass
            return False

    def step(self):
        """执行一个仿真步骤"""
        try:
            if self.simulation_started and self.is_connected():
                traci.simulationStep()
                return True
            return False
        except Exception as e:
            print(f"Error during simulation step: {str(e)}")
            return False

    def get_junction_vehicle_counts(self, junction_id):
        """获取指定路口各方向的交通信息"""
        try:
            if not self.simulation_started or not self.is_connected():
                return None, None
            
            current_time = datetime.datetime.now()
            traffic_data = defaultdict(lambda: {
                'vehicle_count': 0,     # 车辆总数
                'mean_speed': 0,        # 平均速度
                'halting_count': 0,     # 停止的车辆数
                'waiting_time': 0,      # 平均等待时间
            })
            
            junction_info = self.junctions_data.get(junction_id)
            if not junction_info:
                return None, None
            
            for direction, lanes in junction_info["incoming_lanes"].items():
                total_speed = 0
                total_waiting_time = 0
                total_vehicles = 0
                
                for lane in lanes:
                    lane_id = lane['lane_id']
                    try:
                        # 获取车道上的所有车辆ID
                        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                        
                        # 基础统计
                        traffic_data[direction]['vehicle_count'] += len(vehicle_ids)
                        traffic_data[direction]['halting_count'] += traci.lane.getLastStepHaltingNumber(lane_id)
                        
                        # 计算平均速度和等待时间
                        for vehicle_id in vehicle_ids:
                            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                            waiting_time = traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
                            total_speed += vehicle_speed
                            total_waiting_time += waiting_time
                            total_vehicles += 1
                        
                    except traci.exceptions.TraCIException as e:
                        print(f"Error getting data for lane {lane_id}: {str(e)}")
                        continue
                
                # 计算平均值
                if total_vehicles > 0:
                    traffic_data[direction]['mean_speed'] = total_speed / total_vehicles
                    traffic_data[direction]['waiting_time'] = total_waiting_time / total_vehicles
                
                # 更新历史数据
                self.vehicle_counts[junction_id][direction].append(traffic_data[direction]['vehicle_count'])
                self.timestamps[junction_id][direction].append(current_time)
                
                if len(self.vehicle_counts[junction_id][direction]) > 20:
                    self.vehicle_counts[junction_id][direction].pop(0)
                    self.timestamps[junction_id][direction].pop(0)
            
            return dict(traffic_data), self.get_historical_data(junction_id) # {'南北方向直行与右转': {'vehicle_count': 37, 'mean_speed': 1.0774107930032664, 'halting_count': 30, 'waiting_time': 54.67567567567568}, '南北方向左转': {'vehicle_count': 7, 'mean_speed': 0.5072661575047395, 'halting_count': 6, 'waiting_time': 27.285714285714285}, '东向西通行': {'vehicle_count': 0, 'mean_speed': 0, 'halting_count': 0, 'waiting_time': 0}, '西向东通行': {'vehicle_count': 1, 'mean_speed': 20.99044959471956, 'halting_count': 0, 'waiting_time': 0.0}})
        
            
        except Exception as e:
            print(f"Error getting traffic data: {str(e)}")
            return None, None
    
    def get_current_phase(self, junction_id):
        """获取指定路口的当前信号灯相位"""
        try:
            tls_id = junction_id
            current_phase_index = traci.trafficlight.getPhase(tls_id)
            phase_name = self.get_phase_name(tls_id, current_phase_index)
            phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
            remaining_duration = traci.trafficlight.getNextSwitch(tls_id) - traci.simulation.getTime()
            
            return {
                "phase_index": current_phase_index,
                "phase_name": phase_name,
                "total_duration": phase_duration,
                "remaining_duration": remaining_duration
            }
        except Exception as e:
            print(f"Error getting current phase: {str(e)}")
            return None

    def get_phase_name(self, tls_id, phase_index):
        """获取相位的名称描述"""
        try:
            phase_def = traci.trafficlight.getRedYellowGreenState(tls_id)
            # 根据相位状态返回描述性名称
            # 这里需要根据你的信号配置来定制
            phase_names = {
                0: "南北方向直行与右转",
                1: "南北方向左转",
                2: "东西方向直行与右转",
                3: "东西方向左转"
            }
            return phase_names.get(phase_index, f"Phase {phase_index}")
        except:
            return f"Phase {phase_index}"

    def get_simulation_time(self):
        """
        获取当前仿真时间（秒）
        """
        if self.is_connected():
            try:
                # traci.simulation.getTime() 返回当前仿真步数
                # 将步数转换为秒
                return traci.simulation.getTime()
            except Exception as e:
                print(f"获取仿真时间失败: {str(e)}")
                return 0
        return 0

    def close(self):
        """关闭仿真"""
        try:
            if self.is_connected():
                traci.close()
            self.simulation_started = False
            self.warmup_done = False
        except Exception as e:
            print(f"Error closing simulation: {str(e)}")

    def get_phase_info(self, tl_id):
        """
        获取交通信号灯的相位信息
        
        参数:
        tl_id: 交通信号灯ID
        
        返回:
        phase_info: 字典，包含相位数量和当前相位索引.
        示例： {'traffic_light_id': 'J54', 'current_phase_index': 0, 'num_phases': 4, 'phase_durations': [38.0, 30.0, 34.0, 33.0], 'phase_states': ['rrrrGGGGGrrrrrGGGGGr', 'rrrrrrrrrGrrrrrrrrrG', 'rrrrrrrrrrGGGgrrrrrr', 'GGGGrrrrrrrrrrrrrrrr']}
        """
        try:
            # 获取当前相位索引
            current_phase_index = traci.trafficlight.getPhase(tl_id)
            # 获取所有相位的定义
            phase_definitions = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
            # 获取相位数量
            num_phases = len(phase_definitions)
            
            # 获取每个相位的持续时间
            phase_durations = [phase.duration for phase in phase_definitions]
            
            # 获取每个相位的状态（绿灯、红灯等）
            phase_states = [phase.state for phase in phase_definitions]
            
            return {
                'traffic_light_id': tl_id,
                'current_phase_index': current_phase_index,
                'num_phases': num_phases,
                'phase_durations': phase_durations,
                'phase_states': phase_states
            }
            
        except Exception as e:
            print(f"获取交通信号灯相位信息失败: {str(e)}")
            return {}

    def get_phase_controlled_lanes(self, tl_id, phase_index=None):
        """
        获取指定相位控制的进入车道和离开车道
        
        参数:
        tl_id: 交通信号灯ID
        phase_index: 相位索引，如果为None则使用当前相位
        
        返回:
        controlled_lanes: 字典，包含进入车道和离开车道
        """
        try:
            # 如果没有指定相位索引，则使用当前相位
            if phase_index is None:
                phase_index = traci.trafficlight.getPhase(tl_id)
            
            # 获取所有相位的定义
            phase_definitions = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
            if phase_index >= len(phase_definitions):
                print(f"相位索引 {phase_index} 超出范围")
                return {}
            
            # 获取相位状态（绿灯、红灯等）
            phase_state = phase_definitions[phase_index].state
            
            # 获取该相位控制的进入车道和离开车道
            incoming_phase_lanes = []
            outgoing_phase_lanes = []
            
            # 获取该相位的控制链接
            controlled_links = traci.trafficlight.getControlledLinks(tl_id)
            
            for i, state in enumerate(phase_state):
                if i < len(controlled_links):
                    # 如果是绿灯（'G'或'g'），则该链接是当前相位控制的
                    if state in ['G', 'g']:
                        from_lane, to_lane, _ = controlled_links[i][0]
                        incoming_phase_lanes.append(from_lane)
                        outgoing_phase_lanes.append(to_lane)
            
            # 去重
            incoming_phase_lanes = list(set(incoming_phase_lanes))
            outgoing_phase_lanes = list(set(outgoing_phase_lanes))
            
            return {
                'phase_index': phase_index,
                'phase_state': phase_state,
                'incoming_lanes': incoming_phase_lanes,
                'outgoing_lanes': outgoing_phase_lanes
            }
            
        except Exception as e:
            print(f"获取相位控制车道失败: {str(e)}")
            return {}

    def calculate_phase_pressure(self, tl_id, phase_index=None):
        """
        计算指定相位的压力
        压力定义为：某相位的压力 = 该相位进入车道的排队长度 - 该相位离开车道的平均排队长度
        
        参数:
        tl_id: 交通信号灯ID
        phase_index: 相位索引，如果为None则使用当前相位
        
        返回:
        pressure: 相位压力值
        """
        # 获取相位控制的车道
        phase_lanes = self.get_phase_controlled_lanes(tl_id, phase_index)
        if not phase_lanes:
            return 0
        
        # 获取进入车道和离开车道
        incoming_lanes = phase_lanes.get('incoming_lanes', [])
        outgoing_lanes = phase_lanes.get('outgoing_lanes', [])
        
        # 计算进入车道的排队长度总和
        incoming_queue_length = 0
        for lane_id in incoming_lanes:
            # 使用traci.lane.getLastStepHaltingNumber获取停止的车辆数量作为排队长度
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            incoming_queue_length += queue_length
        
        # 计算离开车道的排队长度总和
        outgoing_queue_length = 0
        for lane_id in outgoing_lanes:
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            outgoing_queue_length += queue_length
        
        # 计算离开车道的平均排队长度
        avg_outgoing_queue_length = 0
        if outgoing_lanes:
            avg_outgoing_queue_length = outgoing_queue_length / len(outgoing_lanes)
        
        # 计算相位压力
        pressure = incoming_queue_length - avg_outgoing_queue_length
        
        return {
            'phase_index': phase_lanes.get('phase_index'),
            'incoming_queue_length': incoming_queue_length,
            'outgoing_queue_length': outgoing_queue_length,
            'avg_outgoing_queue_length': avg_outgoing_queue_length,
            'pressure': pressure
        }

    def calculate_all_phases_pressure(self, tl_id):
        """
        计算所有相位的压力
        
        参数:
        tl_id: 交通信号灯ID
        
        返回:
        phase_pressures: 字典，键为相位索引，值为该相位的压力值
        """
        phase_info = self.get_phase_info(tl_id)
        if not phase_info:
            return {}
        
        num_phases = phase_info.get('num_phases', 0)
        phase_pressures = {}
        
        for phase_index in range(num_phases):
            pressure_info = self.calculate_phase_pressure(tl_id, phase_index)
            phase_pressures[phase_index] = pressure_info
        
        return phase_pressures

    def get_max_pressure_phase(self, tl_id):
        """
        获取最大压力相位
        
        参数:
        tl_id: 交通信号灯ID
        
        返回:
        max_pressure_phase: 最大压力相位的索引
        """
        all_pressures = self.calculate_all_phases_pressure(tl_id)
        
        max_pressure = float('-inf')
        max_pressure_phase = None
        
        for phase_index, pressure_info in all_pressures.items():
            if pressure_info['pressure'] > max_pressure:
                max_pressure = pressure_info['pressure']
                max_pressure_phase = phase_index
                
        return max_pressure_phase

    def set_phase_switch(self, tl_id, max_pressure_phase):
        """
        切换相位
        """
        current_time = traci.simulation.getTime()
        current_phase = traci.trafficlight.getPhase(tl_id)
        # 如果当前相位不是最大压力相位，切换到最大压力相位
        if max_pressure_phase is not None and current_phase != max_pressure_phase:
            traci.trafficlight.setPhase(tl_id, max_pressure_phase)
            print(f"时间 {current_time}秒: 从相位 {current_phase} 切换到最大压力相位 {max_pressure_phase}")
        else:
            print(f"时间 {current_time}秒: 保持当前相位 {current_phase}")
        return True

    def get_intersection_metrics(self, tl_id: str, time_window: int = 300) -> Dict[str, float]:
        """
        获取路口指标
        
        参数:
            tl_id: 交通信号灯ID
            time_window: 时间窗口（秒），用于计算时间窗口内的指标
            
        返回:
            路口指标字典，包含：
            - average_saturation: 平均饱和度
            - total_vehicles: 总车辆数
            - average_queue_length: 平均排队长度
            - max_saturation: 最大饱和度
            - max_queue_length: 最大排队长度
            - vehicle_throughput: 车辆通过量（辆/小时）
            - congestion_index: 拥堵指数（0-1）
            - congestion_level: 拥堵等级（非常畅通/基本畅通/轻度拥堵/中度拥堵/严重拥堵）
        """
        if not self.simulation_started:
            print("仿真未启动")
            return {
                'average_saturation': 0.0,
                'total_vehicles': 0,
                'average_queue_length': 0.0,
                'max_saturation': 0.0,
                'max_queue_length': 0.0,
                'vehicle_throughput': 0.0,
                'congestion_index': 0.0,
                'congestion_level': "非常畅通"
            }
            
        # 获取历史数据
        history = self.get_historical_data(tl_id, time_window)
        if not history or not history['timestamps']:
            print("没有历史数据")
            return {
                'average_saturation': 0.0,
                'total_vehicles': 0,
                'average_queue_length': 0.0,
                'max_saturation': 0.0,
                'max_queue_length': 0.0,
                'vehicle_throughput': 0.0,
                'congestion_index': 0.0,
                'congestion_level': "非常畅通"
            }
            
        print(f"获取到 {len(history['timestamps'])} 条历史数据")
        if history['phase_queues']:
            print(f"第一条队列数据示例: {history['phase_queues'][0]}")
            
        # 获取信号灯控制的所有车道，并去重
        controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))
        
        # 初始化统计变量
        total_saturation = 0.0
        total_vehicles = 0
        total_queue_length = 0.0
        max_saturation = 0.0
        max_queue_length = 0.0
        total_delay = 0.0  # 总延误时间
        valid_lanes = 0
        total_steps = 0
        
        # 遍历历史数据
        for i, queues in enumerate(history['phase_queues']):
            step_saturation = 0.0
            step_vehicles = 0
            step_queue_length = 0.0
            step_delay = 0.0  # 该时间点的延误时间
            step_valid_lanes = 0
            
            for lane_id in controlled_lanes:
                try:
                    # 获取车道上的车辆数
                    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                    # 获取车道长度
                    lane_length = 100 #traci.lane.getLength(lane_id)
                    # 获取车道上的排队车辆数
                    halting_vehicles = traci.lane.getLastStepHaltingNumber(lane_id)
                    # 获取车道上的平均延误时间
                    mean_delay = traci.lane.getWaitingTime(lane_id)
                    
                    if lane_length > 0:
                        # 计算饱和度 = (车辆数 * 平均车辆长度 + 排队车辆数 * 平均车辆长度) / 车道长度
                        # 这样能更好地反映拥堵情况
                        saturation = ((vehicle_count + halting_vehicles) * 5) / lane_length
                        step_saturation += saturation
                        
                        # 计算排队长度 = 排队车辆数 * 平均车辆长度
                        queue_length = halting_vehicles * 5
                        step_queue_length += queue_length
                        
                        # 累加延误时间
                        step_delay += mean_delay
                        
                        # 累加车辆数
                        step_vehicles += vehicle_count
                        
                        step_valid_lanes += 1
                except traci.exceptions.TraCIException as e:
                    print(f"获取车道 {lane_id} 数据出错: {e}")
                    continue
            
            if step_valid_lanes > 0:
                # 计算该时间点的平均值
                step_avg_saturation = step_saturation / step_valid_lanes
                step_avg_queue_length = step_queue_length / step_valid_lanes
                step_avg_delay = step_delay / step_valid_lanes
                
                
                # 更新统计值
                total_saturation += step_avg_saturation
                total_vehicles += step_vehicles
                total_queue_length += step_avg_queue_length
                total_delay += step_avg_delay
                
                max_saturation = max(max_saturation, step_avg_saturation)
                max_queue_length = max(max_queue_length, step_avg_queue_length)
                
                total_steps += 1
                valid_lanes = step_valid_lanes
            else:
                print("该时间点没有有效数据")
        
        if total_steps > 0:
            # 计算平均值
            avg_saturation = total_saturation / total_steps
            avg_queue_length = total_queue_length / total_steps
            avg_delay = total_delay / total_steps
            
            # 计算车辆通过量（辆/小时）
            vehicle_throughput = (total_vehicles / time_window) * 3600
            
            # 计算拥堵指数 (0-1)
            # 饱和度权重: 0.4
            # 排队长度权重: 0.3
            # 延误时间权重: 0.3
            congestion_index = (
                0.4 * min(avg_saturation, 1.0) +  # 饱和度
                0.3 * min(avg_queue_length / (valid_lanes * 50), 1.0) +  # 排队长度（假设每条车道最大排队长度为50米）
                0.3 * min(avg_delay / 60, 1.0)  # 延误时间（最大延误60秒）
            )
            
            # 确定拥堵等级
            if congestion_index < 0.3:
                congestion_level = "非常畅通"
            elif congestion_index < 0.5:
                congestion_level = "基本畅通"
            elif congestion_index < 0.7:
                congestion_level = "轻度拥堵"
            elif congestion_index < 0.9:
                congestion_level = "中度拥堵"
            else:
                congestion_level = "严重拥堵"
            
            print(f"\n最终统计:")
            print(f"- 平均饱和度: {avg_saturation:.2f}")
            print(f"- 平均排队长度: {avg_queue_length:.2f}米")
            print(f"- 平均延误: {avg_delay:.1f}秒")
            print(f"- 总车辆数: {total_vehicles}")
            print(f"- 车辆通过量: {vehicle_throughput:.1f}辆/小时")
            print(f"- 拥堵指数: {congestion_index:.2f}")
            print(f"- 拥堵等级: {congestion_level}")
            
            return {
                'average_saturation': avg_saturation,
                'total_vehicles': total_vehicles,
                'average_queue_length': avg_queue_length,
                'max_saturation': max_saturation,
                'max_queue_length': max_queue_length,
                'vehicle_throughput': vehicle_throughput,
                'congestion_index': congestion_index,
                'congestion_level': congestion_level
            }
        else:
            print("没有有效的数据点")
            return {
                'average_saturation': 0.0,
                'total_vehicles': 0,
                'average_queue_length': 0.0,
                'max_saturation': 0.0,
                'max_queue_length': 0.0,
                'vehicle_throughput': 0.0,
                'congestion_index': 0.0,
                'congestion_level': "非常畅通"
            }

    def save_metrics_to_csv(self, tl_id, metrics, csv_file="intersection_metrics.csv"):
        """
        将路口指标数据保存到CSV文件
        """
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写表头
            if not file_exists:
                writer.writerow([
                    "timestamp", "traffic_light_id", "average_saturation", "total_vehicles", "average_queue_length",
                    "max_saturation", "max_queue_length", "vehicle_throughput", "congestion_index", "congestion_level"
                ])
            # 写数据
            writer.writerow([
                datetime.datetime.now().isoformat(),
                tl_id,
                metrics.get('average_saturation', 0),
                metrics.get('total_vehicles', 0),
                metrics.get('average_queue_length', 0),
                metrics.get('max_saturation', 0),
                metrics.get('max_queue_length', 0),
                metrics.get('vehicle_throughput', 0),
                metrics.get('congestion_index', 0),
                metrics.get('congestion_level', "")
            ])
    
    def run_simulation(self, max_steps: int = 3600):
        """
        运行仿真
        
        参数:
            max_steps: 最大仿真步数，默认3600步（1小时）
        """
        if not self.simulation_started:
            print("SUMO未启动")
            return
            
        self.simulation_started = True
        self.simulation_steps = 0
        self.max_steps = max_steps
        
        # 启动仿真线程
        self.simulation_thread = Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
    def _simulation_loop(self):
        """仿真循环"""
        try:
            while self.simulation_started:
                # 检查是否需要重置仿真
                if self.simulation_steps >= self.max_steps:
                    print(f"仿真达到最大步数 {self.max_steps}，重置仿真...")
                    self.reset_simulation()
                    continue
                
                # 执行仿真步
                traci.simulationStep()
                self.simulation_steps += 1
                
                # 收集数据
                self._collect_data()
                
                # 控制仿真速度
                if self.speed_factor > 0:
                    time.sleep(1.0 / self.speed_factor)
                    
        except Exception as e:
            print(f"仿真循环出错: {e}")
            self.simulation_started = False
            
    def reset_simulation(self):
        """重置仿真"""
        try:
            # 保存当前历史数据
            self._save_historical_data()
            
            # 关闭当前仿真
            traci.close()
            
            # 重新启动仿真
            self._start_sumo()
            
            # 重置计数器
            self.simulation_steps = 0
            
            print("仿真已重置")
        except Exception as e:
            print(f"重置仿真出错: {e}")
            self.simulation_started = False
            
    def _collect_data(self):
        """收集仿真数据"""
        try:
            # 获取当前时间
            current_time = datetime.now()
            
            # 收集所有信号灯的数据
            for tl_id in self.traffic_lights:
                # 获取当前相位
                current_phase = traci.trafficlight.getPhase(tl_id)
                
                # 获取相位队列长度
                phase_queues = {}
                for lane_id in traci.trafficlight.getControlledLanes(tl_id):
                    try:
                        phase_queues[lane_id] = traci.lane.getLastStepVehicleNumber(lane_id)
                    except traci.exceptions.TraCIException:
                        continue
                
                # 更新历史数据
                if tl_id not in self.historical_data:
                    self.historical_data[tl_id] = {
                        'timestamps': [],
                        'phase_queues': [],
                        'phases': []
                    }
                
                self.historical_data[tl_id]['timestamps'].append(current_time)
                self.historical_data[tl_id]['phase_queues'].append(phase_queues)
                self.historical_data[tl_id]['phases'].append(current_phase)
                
                # 保持历史数据大小
                if len(self.historical_data[tl_id]['timestamps']) > self.max_history_size:
                    self.historical_data[tl_id]['timestamps'].pop(0)
                    self.historical_data[tl_id]['phase_queues'].pop(0)
                    self.historical_data[tl_id]['phases'].pop(0)
                    
        except Exception as e:
            print(f"收集数据出错: {e}")
            
    def _save_historical_data(self):
        """保存历史数据到文件"""
        try:
            if self.history_file and self.historical_data:
                # 读取现有数据
                existing_data = {}
                if os.path.exists(self.history_file):
                    with open(self.history_file, 'r') as f:
                        existing_data = json.load(f)
                
                # 合并数据
                for tl_id, data in self.historical_data.items():
                    if tl_id not in existing_data:
                        existing_data[tl_id] = {
                            'timestamps': [],
                            'phase_queues': [],
                            'phases': []
                        }
                    
                    # 合并时间戳
                    existing_data[tl_id]['timestamps'].extend(
                        [t.isoformat() for t in data['timestamps']]
                    )
                    existing_data[tl_id]['phase_queues'].extend(data['phase_queues'])
                    existing_data[tl_id]['phases'].extend(data['phases'])
                    
                    # 保持数据大小
                    if len(existing_data[tl_id]['timestamps']) > self.max_history_size:
                        excess = len(existing_data[tl_id]['timestamps']) - self.max_history_size
                        existing_data[tl_id]['timestamps'] = existing_data[tl_id]['timestamps'][excess:]
                        existing_data[tl_id]['phase_queues'] = existing_data[tl_id]['phase_queues'][excess:]
                        existing_data[tl_id]['phases'] = existing_data[tl_id]['phases'][excess:]
                
                # 保存数据
                with open(self.history_file, 'w') as f:
                    json.dump(existing_data, f, indent=2)
                    
        except Exception as e:
            print(f"保存历史数据出错: {e}")

def verify_sumo_config(config_file):
    """验证SUMO配置文件"""
    try:
        if not os.path.exists(config_file):
            return False, "配置文件不存在"
        
        # 读取配置文件内容
        with open(config_file, 'r') as f:
            content = f.read()
        
        # 检查必要的配置项
        required_elements = ['net-file', 'route-files']
        for element in required_elements:
            if element not in content:
                return False, f"缺少必要的配置项: {element}"
        
        return True, "配置文件验证通过"
    except Exception as e:
        return False, f"配置文件验证失败: {str(e)}"

def main():
    """测试函数"""
    current_dir = os.getcwd()
    config_file = os.path.join(current_dir, "osm.sumocfg")
    junctions_file = os.path.join(current_dir, "J54_data.json")
    
    is_valid, message = verify_sumo_config(config_file)
    if not is_valid:
        print(f"错误: {message}")
        return
    
    print("正在启动仿真器...")
    simulator = SUMOSimulator(config_file, junctions_file, gui=True)
    
    try:
        print("正在启动仿真...")
        if simulator.start_simulation():
            print("仿真成功启动")
            
            test_junction = 'J54'
            real_name_junction = {'J54': "倪家桥路与领事馆路交叉口"}
            print(f"\n测试路口: {real_name_junction[test_junction]}")
            
            for i in range(5):
                print(f"\n时间步 {i+1}:")
                traffic_data, _ = simulator.get_junction_vehicle_counts(test_junction)
                phase_info = simulator.get_current_phase(test_junction)
                
                print(traffic_data)
                
                print(f"当前信号相位: {phase_info['phase_name']}")
                print(f"剩余时间: {phase_info['remaining_duration']:.1f}秒")
                
                for direction, data in traffic_data.items():
                    print(f"\n方向: {direction}")
                    print(f"  车辆数量: {data['vehicle_count']} 辆")
                    print(f"  平均速度: {data['mean_speed']:.2f} m/s")
                    print(f"  停止车辆数: {data['halting_count']} 辆")
                    print(f"  平均等待时间: {data['waiting_time']:.1f} 秒")
                
                simulator.step()
                time.sleep(1)
                
    except Exception as e:
        print(f"运行时错误: {str(e)}")
    finally:
        print("正在关闭仿真...")
        simulator.close()
        
        
def test_start_sumo():
    """测试函数"""
    path = r"sumo_llm"
    current_dir = os.getcwd()
    config_file = os.path.join(current_dir, path, "osm.sumocfg")
    junctions_file = os.path.join(current_dir, path, "J54_data.json")
    print(config_file, junctions_file)
    
    is_valid, message = verify_sumo_config(config_file)
    if not is_valid:
        print(f"错误: {message}")
        return
    
    print("正在启动仿真器...")
    simulator = SUMOSimulator(config_file, junctions_file, gui=True)
    
    try:
        if simulator.start_simulation():
            print("仿真成功启动")
            return simulator
                
    except Exception as e:
        print(f"运行时错误: {str(e)}")
    finally:
        print("正在关闭仿真...")
        simulator.close()
        
        
_simulation_manager = None
_simulation_thread = None
_stop_event = Event()

def initialize_sumo(config_file=None, junctions_file=_UNSET, gui=True, history_file=None):
    """
    初始化SUMO模拟器并在后台启动仿真
    Args:
        config_file: SUMO配置文件路径，如果为None则使用默认路径
        junctions_file: 路口数据文件路径；默认使用内置J54_data.json；传入None表示禁用路口数据文件
        gui: 是否使用图形界面
        history_file: 历史数据存储文件路径
    Returns:
        SUMOSimulator实例
    """
    global _simulation_manager, _simulation_thread, _stop_event
    
    # 如果已经有实例在运行，先关闭它
    if _simulation_manager is not None:
        return _simulation_manager
    
    if config_file is None:
        config_file = os.path.join(os.getcwd(), "sumo_llm/osm.sumocfg")
    if junctions_file is _UNSET:
        junctions_file = os.path.join(os.getcwd(), "sumo_llm/J54_data.json")
    if history_file is None:
        history_file = os.path.join(os.path.dirname(config_file), "traffic_history.json")
    
    _stop_event.clear()
    _simulation_manager = SUMOSimulator(config_file, junctions_file, gui, history_file)
    
    if _simulation_manager.start_simulation():
        print("SUMO仿真成功启动")
        
        # 创建并启动仿真线程
        def run_simulation():
            default_tl_id = os.getenv("SUMO_TL_ID", "J54")
            while not _stop_event.is_set():
                if not _simulation_manager.step():
                    break
                    
                # 每10秒收集一次数据
                if _simulation_manager.get_simulation_time() % 10 == 0:
                    _simulation_manager.collect_traffic_data(default_tl_id)
                    metrics = _simulation_manager.get_intersection_metrics(default_tl_id, time_window=3600)
                    _simulation_manager.save_metrics_to_csv(default_tl_id, metrics, csv_file="intersection_metrics.csv")
                    
                time.sleep(1)
        
        _simulation_thread = Thread(target=run_simulation, daemon=True)
        _simulation_thread.start()
        
        return _simulation_manager
    else:
        print("SUMO仿真启动失败")
        _simulation_manager = None
        return None

def stop_simulation():
    """
    停止SUMO仿真
    """
    global _simulation_manager, _simulation_thread, _stop_event
    
    if _simulation_manager is not None:
        _stop_event.set()
        if _simulation_thread is not None:
            _simulation_thread.join(timeout=5)
        _simulation_manager.close()
        _simulation_manager = None
        _simulation_thread = None

def get_simulator():
    """
    获取当前的SUMO模拟器实例
    Returns:
        SUMOSimulator实例，如果未初始化则返回None
    """
    global _simulation_manager
    return _simulation_manager

if __name__ == "__main__":
    try:
        simulator = initialize_sumo()
        if simulator:
            # 主线程保持运行，让仿真在后台进行
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭仿真...")
        stop_simulation()
        print("仿真已关闭")
