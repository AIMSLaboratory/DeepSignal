#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
预测优化算法实现
基于交通流预测来优化信号配时
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque, defaultdict

def _get_simulator():
    from sumo_llm.sumo_simulator import get_simulator as __get_simulator
    return __get_simulator()

class PredictionOptimizer:
    def __init__(self, phases: Dict[int, List[Tuple[int, int]]], 
                 prediction_horizon: int = 10):
        """
        初始化预测优化算法
        
        参数:
            phases: 相位定义字典，格式为 {相位ID: [(进入车道ID, 离开车道ID), ...]}
            prediction_horizon: 预测时间范围（秒）
        """
        self.phases = phases
        self.prediction_horizon = prediction_horizon
        self.min_phase_duration = 15
        self.max_phase_duration = 90
        self.history_size = 300  
        
        # 初始化预测参数
        self.flow_rates = {}  # 各车道的流量率
        self.turn_ratios = {}  # 转向比例
        self.learning_rate = 0.1  # 学习率
        
        # 历史数据缓存
        self.historical_queues = {}  # 车道历史队列长度
        self.historical_phases = {}  # 历史相位数据
        
    def get_historical_data(self, tl_id: str) -> Dict[str, List]:
        """
        从SUMO获取历史数据并进行预处理
        
        参数:
            tl_id: 交通信号灯ID
            
        返回:
            处理后的历史数据字典
        """
        simulator = _get_simulator()
        if simulator is None:
            return {
                'timestamps': [],
                'incoming_queues': [],
                'outgoing_queues': [],
                'phases': []
            }
            
        # 从SUMO获取历史数据
        history = simulator.get_historical_data(tl_id)
        if not history:
            return {
                'timestamps': [],
                'incoming_queues': [],
                'outgoing_queues': [],
                'phases': []
            }
            
        # 更新历史数据缓存
        for phase in self.phases:
            for in_lane, out_lane in self.phases[phase]:
                # 初始化车道历史数据
                if in_lane not in self.historical_queues:
                    self.historical_queues[in_lane] = deque(maxlen=self.history_size)
                if out_lane not in self.historical_queues:
                    self.historical_queues[out_lane] = deque(maxlen=self.history_size)
                
                # 更新历史队列数据
                for queues in history['phase_queues'][-self.history_size:]:
                    if in_lane in queues:
                        self.historical_queues[in_lane].append(queues[in_lane])
                    if out_lane in queues:
                        self.historical_queues[out_lane].append(queues[out_lane])
        
        # 更新历史相位数据
        self.historical_phases = deque(history['phases'][-self.history_size:], maxlen=self.history_size)
        
        # 计算流量率和转向比例
        self.calculate_flow_rates()
        self.calculate_turn_ratios()
        
        return {
            'timestamps': history['timestamps'][-self.history_size:],
            'incoming_queues': history['phase_queues'][-self.history_size:],
            'outgoing_queues': history['phase_queues'][-self.history_size:],
            'phases': history['phases'][-self.history_size:]
        }
    
    def calculate_flow_rates(self) -> None:
        """计算各车道的流量率"""
        for lane, queues in self.historical_queues.items():
            if len(queues) < 2:
                continue
            
            # 计算队列变化率
            changes = []
            for i in range(1, len(queues)):
                change = queues[i] - queues[i-1]
                changes.append(change)
            
            # 使用中位数作为流量率估计
            if changes:
                self.flow_rates[lane] = np.median(changes)
    
    def calculate_turn_ratios(self) -> None:
        """计算转向比例"""
        for phase in self.phases:
            for in_lane, out_lane in self.phases[phase]:
                if (in_lane not in self.historical_queues or 
                    out_lane not in self.historical_queues):
                    continue
                    
                in_queues = list(self.historical_queues[in_lane])
                out_queues = list(self.historical_queues[out_lane])
                
                if len(in_queues) < 2 or len(out_queues) < 2:
                    continue
                
                ratios = []
                for i in range(1, len(in_queues)):
                    in_change = in_queues[i] - in_queues[i-1]
                    out_change = out_queues[i] - out_queues[i-1]
                    
                    if in_change != 0:
                        ratio = out_change / in_change
                        ratios.append(ratio)
                
                if ratios:
                    self.turn_ratios[(in_lane, out_lane)] = np.median(ratios)
    
    def predict_queues(self, current_queues: Dict[int, int],
                      phase_sequence: List[int],
                      phase_durations: List[float]) -> List[Dict[int, int]]:
        """
        预测未来队列长度
        
        参数:
            current_queues: 当前队列长度字典
            phase_sequence: 相位序列
            phase_durations: 相位持续时间序列
        
        返回:
            预测的队列长度序列
        """
        predicted_queues = []
        current_state = current_queues.copy()
        
        # 确保预测时间不超过prediction_horizon
        total_duration = 0
        valid_phases = []
        valid_durations = []
        
        for phase, duration in zip(phase_sequence, phase_durations):
            if total_duration + duration <= self.prediction_horizon:
                valid_phases.append(phase)
                valid_durations.append(duration)
                total_duration += duration
            else:
                # 如果加上这个相位会超过预测范围，则截断
                remaining_time = self.prediction_horizon - total_duration
                if remaining_time > 0:
                    valid_phases.append(phase)
                    valid_durations.append(remaining_time)
                break
            
        for phase, duration in zip(phase_sequence, phase_durations):
            
            phase_movements = self.phases[phase]
            new_state = current_state.copy()
            
            # 使用历史数据计算的流量率和转向比例
            for in_lane, out_lane in phase_movements:
                # 获取车道流量率，如果没有则使用默认值
                flow_rate = self.flow_rates.get(in_lane, 3)  # 默认每秒3辆车
                
                # 获取转向比例，如果没有则使用默认值
                turn_ratio = self.turn_ratios.get((in_lane, out_lane), 0.7)
                
                # 计算移动的车辆数
                moved_vehicles = min(flow_rate * duration, current_state.get(in_lane, 0))
                
                # 更新队列长度
                new_state[in_lane] = max(0, current_state.get(in_lane, 0) - moved_vehicles)
                new_state[out_lane] = current_state.get(out_lane, 0) + moved_vehicles * turn_ratio
            
            predicted_queues.append(new_state)
            current_state = new_state
        
        return predicted_queues
    
    def evaluate_sequence(self, phase_sequence: List[int],
                         phase_durations: List[float],
                         current_queues: Dict[int, int]) -> float:
        """
        评估相位序列的效果
        
        参数:
            phase_sequence: 相位序列
            phase_durations: 相位持续时间序列
            current_queues: 当前队列长度字典
        
        返回:
            序列评分（越低越好）
        """
        predicted_queues = self.predict_queues(current_queues, phase_sequence, phase_durations)
        
        total_score = 0
        weight = 1.0
        
        for queues in predicted_queues:
            # 计算加权队列长度
            queue_score = 0
            for lane, length in queues.items():
                # 根据车道重要性加权
                lane_weight = 1.0
                if 'STRAIGHT' in lane:
                    lane_weight = 1.2  # 直行车道权重更高
                elif 'LEFT' in lane:
                    lane_weight = 0.8  # 左转车道权重较低
                queue_score += length * lane_weight
            
            # 使用指数衰减权重
            total_score += queue_score * weight
            weight *= 0.8  # 更快的衰减
        
        return total_score
    
    def optimize_sequence(self, current_queues: Dict[int, int],
                         current_phase: Optional[int] = None,
                         current_duration: float = 0) -> Tuple[int, float]:
        """
        优化相位序列
        
        参数:
            current_queues: 当前队列长度字典
            current_phase: 当前相位（如果有）
            current_duration: 当前相位持续时间
        
        返回:
            (最优下一相位, 建议持续时间)
        """
        if current_phase is not None and current_duration < self.min_phase_duration:
            return current_phase, self.min_phase_duration - current_duration
        
        # 生成候选序列
        candidates = []
        for phase in self.phases:
            if phase != current_phase:
                # 根据历史数据动态调整持续时间
                base_duration = 30  # 基础持续时间
                if phase in self.flow_rates:
                    # 根据流量率调整持续时间
                    flow_rate = self.flow_rates[phase]
                    duration = min(max(base_duration * (1 + flow_rate/10), 
                                     self.min_phase_duration), 
                                 self.max_phase_duration)
                    candidates.append(([phase], [duration]))
                else:
                    # 尝试不同的持续时间
                    for duration in [15, 30, 45, 60, 75, 90]:
                        candidates.append(([phase], [duration]))
        
        # 评估所有候选序列
        best_score = float('inf')
        best_sequence = None
        best_duration = None
        
        for sequence, durations in candidates:
            score = self.evaluate_sequence(sequence, durations, current_queues)
            if score < best_score:
                best_score = score
                best_sequence = sequence
                best_duration = durations[0]
        
        if best_sequence is None:
            # 如果没有找到更好的序列，继续当前相位
            return current_phase or 0, self.min_phase_duration
        
        return best_sequence[0], best_duration
    
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
        # 获取最优相位和持续时间
        optimal_phase, phase_duration = self.optimize_sequence(
            phase_queues, current_phase, current_duration
        )
        
        return optimal_phase 
