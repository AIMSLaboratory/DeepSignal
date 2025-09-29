#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交通信号控制算法MCP工具服务器
包含三个算法:
1. Max Pressure算法
2. 预测优化算法
3. LLM控制器算法
4. 性能评估器
"""

import os
import sys
import asyncio
from typing import Dict, List, Tuple, Any, Optional
from mcp.server.fastmcp import FastMCP
from max_pressure import MaxPressureAlgorithm

from llm_controller import LLMController


# 添加项目根目录到 Python 路径
print(os.path.abspath(__file__))
print(os.getcwd())
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(current_dir)
sys.path.append(current_dir)

from prediction_optimizer import PredictionOptimizer
from performance_evaluator import PerformanceEvaluator
from sumo_llm.sumo_simulator import get_simulator,initialize_sumo
from datetime import datetime

# 初始化FastMCP服务器
mcp = FastMCP("traffic_control")

# 获取SUMO模拟器实例
print(os.getcwd())
simulator = initialize_sumo()

# 历史数据存储
historical_data = {
    'timestamps': [],
    'phase_queues': [],
    'phases': []
}

async def collect_historical_data():
    """定期收集历史数据"""
    while True:
        try:
            # 获取SUMO模拟器实例
            simulator = get_simulator()
            if simulator is None:
                await asyncio.sleep(10)
                continue
                
            # 获取所有路口的相位队列数据
            tl_id = "J54"  # 这里可以根据需要修改为其他路口ID
            phase_queues = get_phase_queues_from_sumo(tl_id)
            
            # 获取当前相位信息
            phase_info = get_current_phase_info(tl_id)
            current_phase = phase_info.get('phase_info', {}).get('phase_index', 0)
            
            # 保存数据
            historical_data['timestamps'].append(datetime.now().isoformat())
            historical_data['phase_queues'].append(phase_queues)
            historical_data['phases'].append(current_phase)
            
            # 每10秒收集一次数据
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"收集历史数据时发生错误: {str(e)}")
            await asyncio.sleep(10)

@mcp.tool()
def get_historical_data(tl_id: str, time_window: Optional[int] = None) -> Dict[str, Any]:
    """
    获取历史数据
    
    Args:
        tl_id: 交通信号灯ID
        time_window: 时间窗口（秒），None表示使用全部历史数据
    
    Returns:
        历史数据字典
    """
    simulator = get_simulator()
    if simulator is None:
        return {
            "status": "error",
            "message": "SUMO simulator not initialized"
        }
    
    return simulator.get_historical_data(tl_id, time_window)

# 可以添加更多属性
class Phase:
    def __init__(self, movements, min_duration=15, max_duration=90):
        self.movements = movements
        self.min_duration = min_duration
        self.max_duration = max_duration
        
# 可以添加转向限制
class Movement:
    def __init__(self, direction, movement_type, restrictions=None):
        self.direction = direction
        self.movement_type = movement_type
        self.restrictions = restrictions or []

# 示例相位定义
class Movement:
    STRAIGHT = 'STRAIGHT'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    
class Direction:
    NORTH = 'N'
    SOUTH = 'S'
    EAST = 'E'
    WEST = 'W'

def create_phase(movements: List[Dict[str, str]]) -> List[str]:
    """
    创建相位
    Args:
        movements: 相位中包含的转向动作列表
        例如: [
            {'direction': 'N', 'movement': 'STRAIGHT'},
            {'direction': 'S', 'movement': 'STRAIGHT'}
        ]
    """
    return [f"{m['direction']}_{m['movement']}" for m in movements]

# 动态创建相位配置
PHASES = {}

# 示例：创建标准十字路口的四相位配置
PHASES[0] = create_phase([  # 南北直行
    {'direction': Direction.NORTH, 'movement': Movement.STRAIGHT},
    {'direction': Direction.SOUTH, 'movement': Movement.STRAIGHT}
])

PHASES[2] = create_phase([  # 东西直行
    {'direction': Direction.EAST, 'movement': Movement.STRAIGHT},
    {'direction': Direction.WEST, 'movement': Movement.STRAIGHT}
])

PHASES[1] = create_phase([  # 南北左转
    {'direction': Direction.NORTH, 'movement': Movement.LEFT},
    {'direction': Direction.SOUTH, 'movement': Movement.LEFT}
])

PHASES[3] = create_phase([  # 东西左转
    {'direction': Direction.EAST, 'movement': Movement.LEFT},
    {'direction': Direction.WEST, 'movement': Movement.LEFT}
])

# 初始化三个算法实例
max_pressure = MaxPressureAlgorithm(PHASES)
prediction_optimizer = PredictionOptimizer(PHASES)
llm_controller = LLMController(None)  # 这里的None需要替换为实际的LLM客户端

# 初始化性能评估器
performance_evaluator = PerformanceEvaluator()

@mcp.tool()
def get_phase_queues_from_sumo(tl_id: str) -> Dict[str, Dict[str, int]]:
    """
    Args:
        tl_id: 交通信号灯ID
    
    Returns:
        相位队列长度字典，格式为：
        {   "current_phase":{"phase_index":0,"remaining_duration":15},
            "N_STRAIGHT": {"in": 10, "out": 5},
            "S_STRAIGHT": {"in": 8, "out": 3},
            ...
        }
    """
    # 获取所有相位的压力信息
    # 从SUMO获取相位队列数据
    # 0: "南北方向直行与右转",
    # 1: "南北方向左转",
    # 2: "东西方向直行与右转",
    # 3: "东西方向左转"
    simulator = get_simulator()
    phase_pressures = simulator.calculate_all_phases_pressure(tl_id)
    
    current_phase = simulator.get_current_phase(tl_id)
    
    # 获取相位控制的车道信息
    phase_queues = {}
    
    # 遍历每个相位
    for phase_index, pressure_info in phase_pressures.items():
        # 获取该相位控制的车道
        phase_lanes = simulator.get_phase_controlled_lanes(tl_id, phase_index)
        
        # 根据车道方向映射到相位名称
        if phase_index == 0:  # 南北直行
            phase_queues["N_STRAIGHT"] = {
                "in": pressure_info["incoming_queue_length"],
                "out": pressure_info["outgoing_queue_length"]
            }
            phase_queues["S_STRAIGHT"] = {
                "in": pressure_info["incoming_queue_length"],
                "out": pressure_info["outgoing_queue_length"]
            }
        elif phase_index == 2:  # 东西直行
            phase_queues["E_STRAIGHT"] = {
                "in": pressure_info["incoming_queue_length"],
                "out": pressure_info["outgoing_queue_length"]
            }
            phase_queues["W_STRAIGHT"] = {
                "in": pressure_info["incoming_queue_length"],
                "out": pressure_info["outgoing_queue_length"]
            }
        elif phase_index == 1:  # 南北左转
            phase_queues["N_LEFT"] = {
                "in": pressure_info["incoming_queue_length"],
                "out": pressure_info["outgoing_queue_length"]
            }
            phase_queues["S_LEFT"] = {
                "in": pressure_info["incoming_queue_length"],
                "out": pressure_info["outgoing_queue_length"]
            }
        elif phase_index == 3:  # 东西左转
            phase_queues["E_LEFT"] = {
                "in": pressure_info["incoming_queue_length"],
                "out": pressure_info["outgoing_queue_length"]
            }
            phase_queues["W_LEFT"] = {
                "in": pressure_info["incoming_queue_length"],
                "out": pressure_info["outgoing_queue_length"]
            }
    phase_queues["current_phase"] = current_phase
    
    return phase_queues

# @mcp.tool()
# def max_pressure_optimize(phase_queues: Dict[str, Dict[str, int]],
#                         current_phase: Optional[int] = None,
#                         current_duration: float = 0) -> Dict[str, Any]:
#     """使用Max Pressure算法优化信号配时
    
#     Args:
#         phase_queues: 相位队列长度字典，格式为：
#             {
#                 "N_STRAIGHT": {"in": 10, "out": 5},
#                 "S_STRAIGHT": {"in": 8, "out": 3},
#                 ...
#             }
#         current_phase: 当前相位ID
#         current_duration: 当前相位持续时间(秒)
    
#     Returns:
#         优化结果字典
#     """
#     optimal_phase = max_pressure.update(
#         phase_queues,
#         current_phase,
#         current_duration
#     )
    
#     # 记录使用的算法
#     performance_evaluator.set_current_algorithm("max_pressure")
    
#     return {
#         "optimal_phase": optimal_phase,
#         "algorithm": "max_pressure",
#         "status": "success"
#     }

@mcp.tool()
def prediction_optimize(phase_queues: Dict[str, Dict[str, int]],
                      current_phase: Optional[int] = None,
                      current_duration: float = 0) -> Dict[str, Any]:
    """使用预测优化算法优化信号配时
    
    Args:
        phase_queues: 相位队列长度字典
        current_phase: 当前相位ID
        current_duration: 当前相位持续时间(秒)
    
    Returns:
        优化结果字典
    """
    optimal_phase = prediction_optimizer.update(
        phase_queues,
        current_phase,
        current_duration
    )
    
    # 记录使用的算法
    performance_evaluator.set_current_algorithm("prediction_optimizer")
    
    return {
        "optimal_phase": optimal_phase,
        "algorithm": "prediction_optimizer",
        "status": "success"
    }

# @mcp.tool()
# def llm_optimize(phase_queues: Dict[str, Dict[str, int]],
#                 current_phase: Optional[int] = None,
#                 current_duration: float = 0) -> Dict[str, Any]:
#     """使用LLM控制器优化信号配时
    
#     Args:
#         phase_queues: 相位队列长度字典
#         current_phase: 当前相位ID
#         current_duration: 当前相位持续时间(秒)
    
#     Returns:
#         优化结果字典
#     """
#     optimal_phase = llm_controller.update(
#         phase_queues,
#         current_phase,
#         current_duration
#     )
    
#     # 记录使用的算法
#     performance_evaluator.set_current_algorithm("llm_controller")
    
#     return {
#         "optimal_phase": optimal_phase,
#         "algorithm": "llm_controller",
#         "status": "success"
#     }


@mcp.tool()
def get_current_phase_info(tl_id: str) -> Dict[str, Any]:
    """获取当前相位和持续时间信息
    
    Args:
        tl_id: 交通信号灯ID
        
    Returns:
        相位信息字典，包含：
        - phase_index: 当前相位索引
        - phase_name: 相位名称
        - total_duration: 相位总持续时间
        - remaining_duration: 剩余持续时间
    """
    simulator = get_simulator()
    if simulator is None:
        return {
            "status": "error",
            "message": "SUMO simulator not initialized"
        }
    
    # 获取当前相位信息
    phase_info = simulator.get_current_phase(tl_id)
    print("当前相位信息：",phase_info)
    if phase_info is None:
        return {
            "status": "error",
            "message": "Failed to get phase information"
        }
    
    return {
        "status": "success",
        "phase_info": phase_info
    }
    
@mcp.tool()
def set_phase_switch(tl_id: str, max_pressure_phase: int) -> Dict[str, Any]:
    """设置相位切换
    
    Args:
        tl_id: 交通信号灯ID
        max_pressure_phase: 最大压力相位索引
    """
    simulator = get_simulator()
    if simulator is None:
        return {
            "status": "error",
            "message": "SUMO simulator not initialized"
        }
    simulator.set_phase_switch(tl_id, max_pressure_phase)   
    return {
        "status": "success",
        "message": "Phase switch set successfully"
    }

# @mcp.tool()
# def collect_performance_metrics(tl_id: str) -> Dict[str, Any]:
#     """收集交通性能指标
    
#     Args:
#         tl_id: 交通信号灯ID
    
#     Returns:
#         收集的指标数据
#     """
#     return performance_evaluator.collect_metrics(tl_id)

# @mcp.tool()
# def evaluate_algorithm_performance(algorithm_name: str, tl_id: str) -> Dict[str, Any]:
#     """评估指定算法的性能
    
#     Args:
#         algorithm_name: 算法名称，可选值: "max_pressure", "prediction_optimizer"
#         tl_id: 交通信号灯ID
    
#     Returns:
#         评估结果，包含算法得分和详细评分
#     """
#     # 先收集最新指标
#     metrics_result = performance_evaluator.collect_metrics(tl_id)
#     if metrics_result["status"] != "success":
#         return {"status": "error", "message": "无法收集性能指标"}
    
#     # 评估算法性能
#     return performance_evaluator.evaluate_algorithm(algorithm_name, metrics_result.get("metrics"))

# @mcp.tool()
# def get_optimization_suggestion(tl_id: str) -> Dict[str, Any]:
#     """获取优化建议
    
#     Args:
#         tl_id: 交通信号灯ID
    
#     Returns:
#         优化建议，包含最佳算法和相关建议
#     """
#     return performance_evaluator.get_optimization_suggestion(tl_id)

# @mcp.tool()
# def auto_optimize_traffic(tl_id: str) -> Dict[str, Any]:
#     """自动优化交通控制
    
#     Args:
#         tl_id: 交通信号灯ID
    
#     Returns:
#         优化结果
#     """
#     return performance_evaluator.auto_optimize(tl_id, max_pressure, prediction_optimizer)

# @mcp.tool()
# def generate_performance_report(tl_id: str, time_window: Optional[int] = None) -> Dict[str, Any]:
#     """生成性能评估报告
    
#     Args:
#         tl_id: 交通信号灯ID
#         time_window: 时间窗口（秒），None表示使用全部历史数据
    
#     Returns:
#         评估报告，包含平均指标和算法性能统计
#     """
#     # 先确保收集了最新数据
#     performance_evaluator.collect_metrics(tl_id)
    
#     # 生成报告
#     return performance_evaluator.generate_report(time_window)

async def auto_optimize_phase():
    """定时优化相位控制"""
    while True:
        try:
            # 获取SUMO模拟器实例
            simulator = get_simulator()
            if simulator is None:
                await asyncio.sleep(10)
                continue
                
            # 获取所有路口的相位队列数据
            tl_id = "J54"  # 这里可以根据需要修改为其他路口ID
            phase_queues = get_phase_queues_from_sumo(tl_id)
            
            # 获取当前相位信息
            phase_info = get_current_phase_info(tl_id)
            current_phase = phase_info.get('phase_info', {}).get('phase_index', 0)
            current_duration = phase_info.get('phase_info', {}).get('remaining_duration', 0)
            
            # 使用自动优化算法获取最优相位
            optimization_result = auto_optimize_traffic(tl_id)
            
            if optimization_result.get('status') == 'success':
                optimal_phase = optimization_result.get('optimal_phase')
                if optimal_phase != current_phase:
                    # 如果最优相位与当前相位不同，则进行切换
                    set_phase_switch(tl_id, optimal_phase)
                    print(f"相位已从 {current_phase} 切换到 {optimal_phase}")
            
            # 每10秒优化一次
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"自动优化相位时发生错误: {str(e)}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    # 设置服务器端口（可选）
    os.environ['MCP_PORT'] = '8001'
    
    # 创建事件循环
    # try:
    #     loop = asyncio.get_event_loop()
    # except RuntimeError:
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    
    # 启动历史数据收集任务
    # loop.create_task(collect_historical_data())
    
    # # 启动自动优化相位任务
    # loop.create_task(auto_optimize_phase())
    
    # 使用SSE方式运行服务器
    print("Starting traffic control server with SSE transport on port 8001...")
    mcp.run(transport='sse')
