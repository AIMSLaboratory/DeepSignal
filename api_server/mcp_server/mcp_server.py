#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import asyncio
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import gzip
from typing import Dict, List, Tuple, Any, Optional
from mcp.server.fastmcp import FastMCP
from max_pressure import MaxPressureAlgorithm

from llm_controller import LLMController


# 添加项目根目录到 Python 路径
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from prediction_optimizer import PredictionOptimizer
from performance_evaluator import PerformanceEvaluator
from api_server.client.llm_client import LLMClient
from datetime import datetime

# 初始化FastMCP服务器
mcp = FastMCP("traffic_control")

def _sumo_get_simulator():
    from sumo_llm.sumo_simulator import get_simulator as _get_simulator
    return _get_simulator()


def _sumo_initialize_sumo(*, config_file: str, junctions_file: Optional[str], gui: bool, history_file: Optional[str]):
    from sumo_llm.sumo_simulator import initialize_sumo as _initialize_sumo
    return _initialize_sumo(config_file=config_file, junctions_file=junctions_file, gui=gui, history_file=history_file)


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
            simulator = _sumo_get_simulator()
            if simulator is None:
                await asyncio.sleep(10)
                continue
                
            # 获取所有路口的相位队列数据
            tl_id = os.getenv("SUMO_TL_ID", "J54")
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
    simulator = _sumo_get_simulator()
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

# 初始化LLM客户端并传递给LLMController
_llm_controller: Optional[LLMController] = None


def get_llm_controller() -> LLMController:
    global _llm_controller
    if _llm_controller is not None:
        return _llm_controller
    try:
        llm_client = LLMClient(model_type=os.getenv("MODEL_TYPE", "lm-studio"))
        _llm_controller = LLMController(llm_client)
        print("✅ LLM控制器已成功初始化")
    except Exception as e:
        print(f"⚠️ LLM控制器初始化失败: {str(e)}")
        _llm_controller = LLMController(None)  # 降级处理：使用None客户端
    return _llm_controller

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
    simulator = _sumo_get_simulator()
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

# @mcp.tool()
# def prediction_optimize(phase_queues: Dict[str, Dict[str, int]],
#                       current_phase: Optional[int] = None,
#                       current_duration: float = 0) -> Dict[str, Any]:
#     """使用预测优化算法优化信号配时
    
#     Args:
#         phase_queues: 相位队列长度字典
#         current_phase: 当前相位ID
#         current_duration: 当前相位持续时间(秒)
    
#     Returns:
#         优化结果字典
#     """
#     optimal_phase = prediction_optimizer.update(
#         phase_queues,
#         current_phase,
#         current_duration
#     )
    
#     # 记录使用的算法
#     performance_evaluator.set_current_algorithm("prediction_optimizer")
    
#     return {
#         "optimal_phase": optimal_phase,
#         "algorithm": "prediction_optimizer",
#         "status": "success"
#     }

@mcp.tool()
def llm_optimize(phase_queues: Dict[str, Dict[str, int]],
                current_phase: Optional[int] = None,
                current_duration: float = 0) -> Dict[str, Any]:
    """使用LLM控制器优化信号配时
    
    Args:
        phase_queues: 相位队列长度字典
        current_phase: 当前相位ID
        current_duration: 当前相位持续时间(秒)
    
    Returns:
        优化结果字典
    """
    llm_controller = get_llm_controller()
    optimal_phase = llm_controller.update(
        phase_queues,
        current_phase,
        current_duration
    )
    
    # 记录使用的算法
    performance_evaluator.set_current_algorithm("llm_controller")
    
    return {
        "optimal_phase": optimal_phase,
        "algorithm": "llm_controller",
        "status": "success"
    }


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
    simulator = _sumo_get_simulator()
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
    simulator = _sumo_get_simulator()
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

@mcp.tool()
def auto_optimize_and_switch_phase(tl_id: str) -> Dict[str, Any]:
    """自动优化并切换相位 - 完整的信号优化工作流
    
    这是一个高级工具，集成了以下步骤：
    1. 获取当前相位信息
    2. 获取所有相位的队列数据
    3. 使用LLM控制器进行优化
    4. 如果最优相位不同，则自动切换
    
    Args:
        tl_id: 交通信号灯ID
    
    Returns:
        优化和切换结果字典
    """
    try:
        # 步骤1：获取当前相位信息
        phase_info = get_current_phase_info(tl_id)
        if phase_info.get("status") != "success":
            return {
                "status": "error",
                "message": f"Failed to get current phase info: {phase_info.get('message')}",
                "action_taken": False
            }
        
        current_phase_data = phase_info.get('phase_info', {})
        current_phase = current_phase_data.get('phase_index', 0)
        remaining_duration = current_phase_data.get('remaining_duration', 0)
        
        # 步骤2：获取所有相位的队列数据
        phase_queues = get_phase_queues_from_sumo(tl_id)
        if not phase_queues:
            return {
                "status": "error",
                "message": "Failed to get phase queues",
                "action_taken": False
            }
        
        # 步骤3：使用LLM控制器进行优化
        optimization_result = llm_optimize(phase_queues, current_phase, remaining_duration)
        if optimization_result.get("status") != "success":
            return {
                "status": "error",
                "message": f"LLM optimization failed: {optimization_result}",
                "current_phase": current_phase,
                "action_taken": False
            }
        
        optimal_phase = optimization_result.get("optimal_phase")
        
        # 步骤4：比较并执行切换
        action_taken = False
        message = ""
        
        if optimal_phase != current_phase:
            # 执行相位切换
            switch_result = set_phase_switch(tl_id, optimal_phase)
            if switch_result.get("status") == "success":
                action_taken = True
                message = f"相位已从 {current_phase} 成功切换到 {optimal_phase}"
                print(f"✅ {message}")
            else:
                message = f"尝试切换相位失败: {switch_result.get('message')}"
                print(f"❌ {message}")
        else:
            message = f"当前相位 {current_phase} 已经是最优相位，无需切换"
            print(f"ℹ️  {message}")
        
        return {
            "status": "success",
            "current_phase": current_phase,
            "optimal_phase": optimal_phase,
            "action_taken": action_taken,
            "message": message,
            "phase_queues": phase_queues,
            "optimization_details": optimization_result,
            "algorithm": "llm_controller"
        }
        
    except Exception as e:
        print(f"自动优化和切换相位时发生错误: {str(e)}")
        return {
            "status": "error",
            "message": f"Auto optimize and switch failed: {str(e)}",
            "action_taken": False
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
    """定时优化相位控制 - 增强版，包含详细日志和验证"""
    optimization_count = 0
    
    while True:
        try:
            optimization_count += 1
            start_time = datetime.now()
            
            # 获取SUMO模拟器实例
            simulator = _sumo_get_simulator()
            if simulator is None:
                print("⚠️ SUMO模拟器未初始化，等待...")
                await asyncio.sleep(10)
                continue
            
            # 获取所有路口的相位队列数据
            tl_id = os.getenv("SUMO_TL_ID", "J54")
            
            print(f"\n{'='*60}")
            print(f"第 {optimization_count} 次优化 ({start_time.strftime('%H:%M:%S')})")
            print(f"{'='*60}")
            
            # 步骤1：获取当前相位信息
            phase_info = get_current_phase_info(tl_id)
            if phase_info.get("status") != "success":
                print(f"❌ 获取当前相位失败: {phase_info.get('message')}")
                await asyncio.sleep(10)
                continue
            
            current_phase_data = phase_info.get('phase_info', {})
            current_phase = current_phase_data.get('phase_index', 0)
            remaining_duration = current_phase_data.get('remaining_duration', 0)
            
            print(f"📊 [步骤1] 当前相位: {current_phase}, 剩余时间: {remaining_duration}秒")
            
            # 步骤2：获取所有相位的队列数据
            phase_queues = get_phase_queues_from_sumo(tl_id)
            if not phase_queues:
                print("❌ 获取相位队列数据失败")
                await asyncio.sleep(10)
                continue
            
            print(f"🚗 [步骤2] 各相位队列情况:")
            for phase_name, queue_data in phase_queues.items():
                if phase_name != "current_phase" and isinstance(queue_data, dict):
                    print(f"   - {phase_name}: 进入={queue_data.get('in', 0)}, 离开={queue_data.get('out', 0)}")
            
            # 步骤3：使用LLM控制器进行优化
            optimization_result = llm_optimize(phase_queues, current_phase, remaining_duration)
            
            if optimization_result.get("status") != "success":
                print(f"❌ LLM优化失败: {optimization_result}")
                await asyncio.sleep(10)
                continue
            
            optimal_phase = optimization_result.get("optimal_phase")
            print(f"💡 [步骤3] LLM建议相位: {optimal_phase}")
            
            # 步骤4：比较并执行切换
            if optimal_phase != current_phase:
                print(f"⚡ [步骤4] 执行切换: {current_phase} → {optimal_phase}")
                
                # 执行相位切换
                switch_result = set_phase_switch(tl_id, optimal_phase)
                
                if switch_result.get("status") == "success":
                    # 等待切换完成
                    await asyncio.sleep(1)
                    
                    # 验证切换结果
                    new_phase_info = get_current_phase_info(tl_id)
                    if new_phase_info.get("status") == "success":
                        actual_phase = new_phase_info.get('phase_info', {}).get('phase_index')
                        
                        if actual_phase == optimal_phase:
                            print(f"✅ [验证成功] 相位已按LLM建议从 {current_phase} 切换到 {actual_phase}")
                        else:
                            print(f"⚠️ [验证失败] LLM建议切换到 {optimal_phase}，但实际相位为 {actual_phase}")
                    else:
                        print(f"⚠️ [验证失败] 无法获取切换后的相位信息")
                else:
                    print(f"❌ [切换失败] {switch_result.get('message')}")
            else:
                print(f"ℹ️  [无需切换] 当前相位 {current_phase} 已是LLM建议的最优相位")
            
            # 计算耗时
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"⏱️  [耗时] 本次优化用时: {duration:.2f}秒")
            print(f"{'='*60}\n")
            
            # 每10秒优化一次
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"❌ 自动优化相位时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(10)

def run_auto_optimize_in_thread():
    """在独立线程中运行自动优化任务"""
    import threading
    import time
    
    def run_async_loop():
        # 创建新的事件循环用于后台任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 添加延迟确保SUMO和其他服务完全启动
        time.sleep(2)
        
        # 创建并运行自动优化任务
        loop.create_task(auto_optimize_phase())
        
        # 持续运行事件循环
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            print("\n⚠️ 自动优化任务被中断")
        finally:
            loop.close()
    
    # 启动后台线程
    thread = threading.Thread(target=run_async_loop, daemon=True)
    thread.start()
    print("✅ 自动优化后台线程已启动")


def _list_scenarios() -> List[str]:
    scenarios_dir = REPO_ROOT / "scenarios"
    if not scenarios_dir.exists():
        return []
    return sorted([p.name for p in scenarios_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])

def _parse_sumocfg_net_file(sumocfg_path: Path) -> Path:
    """
    Extract net-file from a .sumocfg and return an absolute path.
    Supports common SUMO config structures like:
      <input><net-file value="..."/></input>
    """
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()

    net_file_value: Optional[str] = None
    for el in root.iter():
        # SUMO uses tags like "net-file" (with hyphen)
        if el.tag.endswith("net-file"):
            net_file_value = el.attrib.get("value") or (el.text.strip() if el.text else None)
            if net_file_value:
                break

    if not net_file_value:
        raise ValueError(f"Could not find <net-file> in {sumocfg_path}")

    net_path = Path(net_file_value).expanduser()
    if not net_path.is_absolute():
        net_path = (sumocfg_path.parent / net_path).resolve()
    else:
        net_path = net_path.resolve()

    return net_path


def _list_traffic_light_ids(sumocfg_path: Path) -> List[str]:
    net_path = _parse_sumocfg_net_file(sumocfg_path)
    if not net_path.exists():
        raise FileNotFoundError(f"net-file not found: {net_path}")

    if net_path.suffix == ".gz":
        with gzip.open(net_path, "rb") as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(net_path)
    root = tree.getroot()

    tl_ids: List[str] = []
    for el in root.iter():
        if el.tag.endswith("tlLogic"):
            tl_id = el.attrib.get("id")
            if tl_id:
                tl_ids.append(tl_id)

    # unique + sorted
    return sorted(set(tl_ids))


def _resolve_sumocfg_path(scenario: Optional[str], sumocfg: Optional[str]) -> Path:
    if sumocfg:
        return Path(sumocfg).expanduser().resolve()

    if scenario:
        scenario_path = Path(scenario).expanduser()
        if scenario_path.is_absolute() or ("/" in scenario) or ("\\" in scenario) or scenario.startswith("."):
            scenario_dir = scenario_path.resolve()
        else:
            scenario_dir = (REPO_ROOT / "scenarios" / scenario).resolve()

        if not scenario_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        candidates = sorted(scenario_dir.glob("*.sumocfg"))
        if not candidates:
            raise FileNotFoundError(f"No .sumocfg found under: {scenario_dir}")
        if len(candidates) == 1:
            return candidates[0].resolve()

        preferred_names = ["run.sumocfg", "osm.sumocfg", f"{scenario_dir.name}.sumocfg"]
        for name in preferred_names:
            p = scenario_dir / name
            if p.exists():
                return p.resolve()

        raise ValueError(
            "Multiple .sumocfg files found; please pass --sumocfg explicitly: "
            + ", ".join([c.name for c in candidates])
        )

    return (REPO_ROOT / "sumo_llm" / "osm.sumocfg").resolve()


def _resolve_junctions_file(sumocfg_path: Path, junctions_file: Optional[str]) -> Optional[Path]:
    if junctions_file:
        return Path(junctions_file).expanduser().resolve()

    # 默认内置J54配置仅适用于项目自带的 sumo_llm 场景
    try:
        if sumocfg_path.samefile((REPO_ROOT / "sumo_llm" / "osm.sumocfg").resolve()):
            return (REPO_ROOT / "sumo_llm" / "J54_data.json").resolve()
    except Exception:
        pass

    scenario_dir = sumocfg_path.parent
    candidates = []
    candidates.extend(sorted(scenario_dir.glob("J*_data.json")))
    candidates.extend(sorted(scenario_dir.glob("*_data.json")))
    candidates.extend(sorted(scenario_dir.glob("*junction*.json")))

    # 去重
    seen = set()
    uniq: List[Path] = []
    for c in candidates:
        rp = c.resolve()
        if rp not in seen:
            uniq.append(rp)
            seen.add(rp)

    if not uniq:
        return None
    if len(uniq) == 1:
        return uniq[0]

    # 多个候选时优先 J54_data.json
    for c in uniq:
        if c.name == "J54_data.json":
            return c

    return None


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP server for SUMO traffic control")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios under ./scenarios")
    parser.add_argument("--list-tl-ids", action="store_true", help="List traffic light ids (TL IDs) in the selected scenario/sumocfg")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name under ./scenarios (or a path to a scenario dir)")
    parser.add_argument("--sumocfg", type=str, default=None, help="Explicit path to a .sumocfg file (overrides --scenario)")
    parser.add_argument("--junctions-file", type=str, default=None, help="Path to junctions JSON file (optional)")
    parser.add_argument("--tl-id", type=str, default=os.getenv("SUMO_TL_ID", "J54"), help="Traffic light id (default: J54)")
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8001")), help="MCP server port (default: 8001)")
    parser.add_argument("--nogui", action="store_true", help="Run SUMO without GUI")
    parser.add_argument("--no-auto-optimize", action="store_true", help="Disable 10s background auto-optimization loop")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    if args.list_scenarios:
        for name in _list_scenarios():
            print(name)
        return 0

    sumocfg_path = _resolve_sumocfg_path(args.scenario, args.sumocfg)

    if args.list_tl_ids:
        for tl_id in _list_traffic_light_ids(sumocfg_path):
            print(tl_id)
        return 0

    junctions_path = _resolve_junctions_file(sumocfg_path, args.junctions_file)

    os.environ["MCP_PORT"] = str(args.port)
    os.environ["SUMO_TL_ID"] = args.tl_id

    print("=" * 60)
    print("🚦 交通信号控制系统启动")
    print("=" * 60)
    print(f"📁 SUMO config: {sumocfg_path}")
    print(f"🚦 TL ID: {args.tl_id}")
    print(f"🖥️  GUI: {not args.nogui}")
    if junctions_path:
        print(f"🗺️  Junctions file: {junctions_path}")
    else:
        print("🗺️  Junctions file: (disabled / not found)")
    print("=" * 60)

    _sumo_initialize_sumo(
        config_file=str(sumocfg_path),
        junctions_file=str(junctions_path) if junctions_path else None,
        gui=not args.nogui,
        history_file=None,
    )

    print("✅ SUMO仿真已初始化")
    print("✅ MCP服务器准备就绪")
    print("✅ LLM控制器已加载")
    if not args.no_auto_optimize:
        print("⏰ 自动优化任务准备启动 (每10秒执行)")
    print("=" * 60)

    if not args.no_auto_optimize:
        run_auto_optimize_in_thread()

    print(f"\n🌐 Starting MCP server with SSE transport on port {args.port}...")
    print("📡 LLM将每10秒自动分析交通状态并优化信号相位\n")
    mcp.run(transport="sse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
