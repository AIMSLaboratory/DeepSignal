#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Max Pressure 算法实现
基于车道压力（队列长度）来优化信号配时
"""

from typing import Dict, List, Tuple, Optional

class MaxPressureAlgorithm:
    def __init__(self, phases: Dict[int, List[str]]):
        """
        初始化 Max Pressure 算法
        
        参数:
            phases: 相位定义字典，格式为 {相位ID: [movement_id, ...]}
            movement_id格式为: "方向_转向类型"，例如 "N_STRAIGHT"
        """
        self.phases = phases
        self.min_phase_duration = 15  # 最小相位持续时间（秒）
        self.max_phase_duration = 90  # 最大相位持续时间（秒）
    
    def calculate_pressure(self, phase: List[str], 
                         phase_queues: Dict[str, Dict[str, int]]) -> float:
        """
        计算某个相位的压力值
        
        参数:
            phase: 相位定义，包含移动方向列表，如 ["N_STRAIGHT", "S_STRAIGHT"]
            phase_queues: 相位队列长度字典 {movement_id: {'in': 进口队列长度, 'out': 出口队列长度}}
            例如：{
                'N_STRAIGHT': {'in': 10, 'out': 5},
                'S_STRAIGHT': {'in': 8, 'out': 3}
            }
        
        返回:
            相位压力值
        """
        total_pressure = 0
        
        for movement in phase:
            if movement in phase_queues:
                queues = phase_queues[movement]
                in_queue = queues.get('in', 0)
                out_queue = queues.get('out', 0)
                
                # 计算压力值（进口队列长度与出口队列长度的差值）
                pressure = max(0, in_queue - out_queue)
                total_pressure += pressure
        
        return total_pressure
    
    def get_optimal_phase(self, phase_queues: Dict[str, Dict[str, int]],
                         current_phase: Optional[int] = None,
                         current_duration: float = 0) -> Tuple[int, float]:
        """
        获取最优相位
        
        参数:
            phase_queues: 相位队列长度字典
            current_phase: 当前相位ID（如果有）
            current_duration: 当前相位持续时间
        
        返回:
            (最优相位ID, 建议持续时间)
        """
        # 如果当前相位未达到最小持续时间，继续当前相位
        if current_phase is not None and current_duration < self.min_phase_duration:
            return current_phase, self.min_phase_duration - current_duration
        
        # 计算每个相位的压力值
        phase_pressures = {}
        for phase_id, phase_def in self.phases.items():
            pressure = self.calculate_pressure(phase_def, phase_queues)
            phase_pressures[phase_id] = pressure
        
        # 找出压力值最大的相位
        optimal_phase = max(phase_pressures.items(), key=lambda x: x[1])[0]
        
        # 如果最优相位与当前相位相同且未超过最大持续时间，继续当前相位
        if optimal_phase == current_phase and current_duration < self.max_phase_duration:
            return optimal_phase, self.max_phase_duration - current_duration
        
        # 否则切换到新相位
        return optimal_phase, self.min_phase_duration

    def update(self, phase_queues: Dict[str, Dict[str, int]],
              current_phase: Optional[int] = None,
              current_duration: float = 0) -> int:
        """
        更新并返回最优相位（简化接口）
        
        参数:
            phase_queues: 相位队列长度字典
            current_phase: 当前相位ID（如果有）
            current_duration: 当前相位持续时间
        
        返回:
            最优相位ID
        """
        optimal_phase, _ = self.get_optimal_phase(
            phase_queues, current_phase, current_duration
        )
        return optimal_phase
    
    
    # def calculate_pressure(self, phase: List[Tuple[int, int]], 
    #                   incoming_queues: Dict[int, int],
    #                   outgoing_queues: Dict[int, int]) -> float:
    #     total_pressure = 0
    #     for in_lane, out_lane in phase:
    #         # 获取队列长度
    #         in_queue = incoming_queues.get(in_lane, 0)
    #         out_queue = outgoing_queues.get(out_lane, 0)
            
    #         # 考虑车道容量
    #         in_capacity = self.lane_capacities.get(in_lane, 100)
    #         out_capacity = self.lane_capacities.get(out_lane, 100)
            
    #         # 计算标准化的队列长度
    #         in_pressure = in_queue / in_capacity
    #         out_pressure = out_queue / out_capacity
            
    #         # 考虑转向系数
    #         turn_factor = self.turn_ratios.get((in_lane, out_lane), 1.0)
            
    #         # 计算综合压力
    #         pressure = (in_pressure + out_pressure) * turn_factor
            
    #         # 考虑饱和度
    #         saturation_rate = self.get_saturation_rate(in_lane)
    #         pressure *= saturation_rate
            
    #         total_pressure += pressure
        
    #     return total_pressure
    
    # def get_optimal_phase(self, incoming_queues: Dict[int, int],
    #                  outgoing_queues: Dict[int, int],
    #                  current_phase: Optional[int] = None,
    #                  current_duration: float = 0) -> Tuple[int, float]:
    
    #     # 计算切换损耗
    #     switch_cost = self.calculate_switch_cost(current_phase)
        
    #     # 计算每个相位的压力值
    #     phase_pressures = {}
    #     for phase_id, phase_def in self.phases.items():
    #         base_pressure = self.calculate_pressure(phase_def, incoming_queues, outgoing_queues)
            
    #         # 考虑相位切换损耗
    #         if phase_id != current_phase:
    #             base_pressure -= switch_cost
                
    #         # 考虑历史表现
    #         historical_performance = self.get_historical_performance(phase_id)
            
    #         # 计算最终压力值
    #         phase_pressures[phase_id] = base_pressure * historical_performance
        
    #     # 找出压力值最大的相位
    #     optimal_phase = max(phase_pressures.items(), key=lambda x: x[1])[0]
        
    #     # 动态计算绿灯时间
    #     green_time = self.calculate_green_time(
    #         phase_pressures[optimal_phase],
    #         current_phase,
    #         current_duration
    #     )
        
    #     return optimal_phase, green_time