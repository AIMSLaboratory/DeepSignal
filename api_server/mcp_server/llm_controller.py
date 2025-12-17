#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大语言模型控制器实现 - 适配微调模型格式
使用LLM进行智能信号控制决策

基于微调数据格式：
instruction: 你是一位交通管理专家。你可以运用你的交通常识知识来解决交通信号控制任务。
             根据给定的交通场景和状态，预测下一个信号相位。
             你必须直接回答：下一个信号相位是={你预测的相位}
input: 路口场景描述 + 交通状态描述
output: 下一个信号相位：X
"""

import json
import re
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

class LLMController:
    def __init__(self, client: Any, junction_config: Optional[Dict] = None):
        """
        初始化LLM控制器
        
        参数:
            client: LLM客户端（支持OpenAI、Claude等）
            junction_config: 路口配置信息（来自J54_data.json）
        """
        self.client = client
        self.junction_config = junction_config or {}
        self.min_phase_duration = 15
        self.max_phase_duration = 90
        
        # 相位定义（J54路口）
        self.phase_definitions = {
            0: "南北方向直行与右转",
            1: "南北方向左转",
            2: "东西方向直行与右转",
            3: "东西方向左转"
        }
        
        # 相位控制的车道信息（根据J54_data.json实际配置）
        self.phase_lane_info = {
            0: {  # 南北方向直行与右转
                "description": "南北方向直行与右转",
                "lane_count": 8,
                "lanes": [
                    "136004889.148_0", "136004889.148_1", "136004889.148_2", "136004889.148_3",  # 南向
                    "-136004889.221.207_0", "-136004889.221.207_1", "-136004889.221.207_2", "-136004889.221.207_3"  # 北向
                ],
                "directions": ["南向直行右转", "北向直行右转"]
            },
            1: {  # 南北方向左转
                "description": "南北方向左转",
                "lane_count": 2,
                "lanes": [
                    "-136004889.221.207_4",  # 北向左转
                    "136004889.148_4"  # 南向左转
                ],
                "directions": ["北向左转", "南向左转"]
            },
            2: {  # 东西直行
                "description": "东西方向直行与右转",
                "lane_count": 4,
                "lanes": [
                    "37132266#4_0", "37132266#4_1",  # 东向
                    "-184446506#4_0", "-184446506#4_1"  # 西向
                ],
                "directions": ["东向直行右转", "西向直行右转"]
            },
            3: {  # 东西方向左转
                "description": "东西方向左转",
                "lane_count": 2,
                "lanes": [
                    "-184446506#4_2",  # 西向左转
                    "37132266#4_2"  # 东向左转
                ],
                "directions": ["西向左转", "东向左转"]
            }
        }
    
    def _generate_junction_description(self) -> str:
        """
        生成路口场景描述（符合微调数据格式）
        
        返回:
            路口场景描述文本
        """
        # 相位列表
        phases = list(self.phase_definitions.keys())
        phase_count = len(phases)
        
        # 计算总车道数量
        total_lane_count = sum(info["lane_count"] for info in self.phase_lane_info.values())
        
        # 生成相位-车道控制关系描述
        phase_lane_desc = []
        for phase_id, info in self.phase_lane_info.items():
            lane_count = info["lane_count"]
            directions = "、".join(info["directions"])
            phase_lane_desc.append(
                f"相位{phase_id}（{info['description']}）控制{lane_count}条车道，包括{directions}"
            )
        
        # 各相位的可观测范围（根据J54实际数据）
        phase_ranges = {
            0: 143.31,  # 南北方向，取较大的北向车道长度
            1: 143.31,  # 南北方向左转
            2: 392.32,  # 东西方向，取较大的东向车道长度
            3: 392.32   # 东西方向左转
        }
        
        range_desc = []
        for phase_id, range_m in phase_ranges.items():
            range_desc.append(f"相位{phase_id}的可观测范围为{range_m:.1f}米")
        
        description = (
            f"路口场景描述：该路口（J54）有{phase_count}个相位，分别是{phases}，"
            f"共有{total_lane_count}条进口车道。"
            f"{'; '.join(phase_lane_desc)}。"
            f"{'; '.join(range_desc)}。"
        )
        
        return description
    
    def _generate_traffic_state(self, phase_queues: Dict[str, Any], 
                                current_phase: int, 
                                current_duration: float) -> str:
        """
        生成实时交通状态描述（符合微调数据格式）
        
        参数:
            phase_queues: 相位队列数据
            current_phase: 当前相位
            current_duration: 当前相位持续时间
        
        返回:
            交通状态描述文本
        """
        # 计算各相位的统计数据
        phase_stats = {}
        
        for phase_id in self.phase_definitions.keys():
            # 获取该相位控制的车道的数据
            in_queue = 0
            out_queue = 0
            total_speed = 0
            total_distance = 0
            lane_count = 0
            
            # 遍历phase_queues，根据相位匹配数据
            for key, value in phase_queues.items():
                if key == "current_phase":
                    continue
                
                # 根据key判断属于哪个相位（简化逻辑）
                if phase_id == 0 and ("N_STRAIGHT" in key or "S_STRAIGHT" in key):
                    in_queue += value.get("in", 0)
                    out_queue += value.get("out", 0)
                    lane_count += 1
                elif phase_id == 1 and ("N_LEFT" in key or "S_LEFT" in key):
                    in_queue += value.get("in", 0)
                    out_queue += value.get("out", 0)
                    lane_count += 1
                elif phase_id == 2 and ("E_STRAIGHT" in key or "W_STRAIGHT" in key):
                    in_queue += value.get("in", 0)
                    out_queue += value.get("out", 0)
                    lane_count += 1
                elif phase_id == 3 and ("E_LEFT" in key or "W_LEFT" in key):
                    in_queue += value.get("in", 0)
                    out_queue += value.get("out", 0)
                    lane_count += 1
            
            # 计算平均值
            avg_vehicles = in_queue / max(lane_count, 1)
            avg_queue = out_queue / max(lane_count, 1)
            avg_speed = 0.5  # 简化处理，实际应从SUMO获取
            avg_distance = 80.0 + phase_id * 10  # 简化处理
            
            phase_stats[phase_id] = {
                "avg_vehicles": avg_vehicles,
                "avg_queue": avg_queue,
                "avg_speed": avg_speed,
                "avg_distance": avg_distance
            }
        
        # 生成描述文本
        state_lines = [
            f"交通状态描述：目前该交叉口的当前相位为{current_phase}，当前相位持续时间为{int(current_duration)}。"
        ]
        
        for phase_id, stats in phase_stats.items():
            state_lines.append(
                f"相位({phase_id})控制的车道的平均车辆数量为{stats['avg_vehicles']:.2f}，"
                f"排队车辆为{stats['avg_queue']:.2f}，"
                f"平均车速为{stats['avg_speed']:.2f}m/s，"
                f"车辆到路口的平均距离为{stats['avg_distance']:.2f}米。"
            )
        
        return "\n".join(state_lines)
    
    def _parse_llm_response(self, response_text: str) -> Optional[int]:
        """
        解析LLM响应，提取相位编号
        
        参数:
            response_text: LLM返回的文本
        
        返回:
            相位编号，如果解析失败返回None
        """
        # 尝试匹配多种格式
        patterns = [
            r'下一个信号相位[是为：:=]+\s*(\d+)',  # 下一个信号相位是=2 或 下一个信号相位：2
            r'下一个信号相位\s*(\d+)',  # 下一个信号相位2
            r'相位[是为：:=]+\s*(\d+)',  # 相位是2
            r'切换到相位\s*(\d+)',  # 切换到相位2
            r'建议相位\s*(\d+)',  # 建议相位2
            r'[选择建议]\s*相位\s*(\d+)',  # 选择相位2
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text)
            if match:
                phase = int(match.group(1))
                # 验证相位编号有效性（0-3）
                if 0 <= phase <= 3:
                    return phase
        
        # 如果都匹配失败，尝试提取最后一个数字
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            phase = int(numbers[-1])
            if 0 <= phase <= 3:
                return phase
        
        return None
    
    def update(self, phase_queues: Dict[str, Any],
              current_phase: Optional[int] = None,
              current_duration: float = 0) -> int:
        """
        更新并返回最优相位（适配微调模型格式）
        
        参数:
            phase_queues: 相位队列数据字典，包含current_phase信息
            current_phase: 当前相位ID（如果phase_queues中没有则使用此参数）
            current_duration: 当前相位持续时间
        
        返回:
            最优相位ID
        """
        try:
            # 检查客户端是否初始化
            if self.client is None:
                print("⚠️ LLM客户端未初始化，返回当前相位")
                return current_phase or 0
            
            # 从phase_queues中提取当前相位信息
            if "current_phase" in phase_queues:
                phase_info = phase_queues["current_phase"]
                if isinstance(phase_info, dict):
                    current_phase = phase_info.get("phase_index", current_phase or 0)
                    current_duration = phase_info.get("remaining_duration", current_duration)
            
            current_phase = current_phase or 0
            
            # 生成路口场景描述
            junction_desc = self._generate_junction_description()
            
            # 生成交通状态描述
            traffic_state = self._generate_traffic_state(phase_queues, current_phase, current_duration)
            
            # 构建完整的instruction和input（符合微调格式）
            instruction = (
                "你是一位交通管理专家。你可以运用你的交通常识知识来解决交通信号控制任务。"
                "根据给定的交通场景和状态，预测下一个信号相位。"
                "你必须直接回答：下一个信号相位是={你预测的相位}"
            )
            
            input_text = f"{junction_desc}\n{traffic_state}"
            
            # 打印LLM输入（用于调试）
            print(f"🤖 [LLM输入-场景] {junction_desc}")
            print(f"🤖 [LLM输入-状态] {traffic_state}")
            
            # 调用LLM
            try:
                response = self.client.chat(
                    messages=[{
                        "role": "user",
                        "content": f"{instruction}\n\n{input_text}"
                    }],
                    temperature=0.3,
                    max_tokens=100
                )
                response_text = response.get('content', '')
                    
            except Exception as e:
                print(f"⚠️ LLM调用失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return current_phase
            
            print(f"🤖 [LLM响应] {response_text}")
            
            # 解析LLM响应
            optimal_phase = self._parse_llm_response(response_text)
            
            if optimal_phase is not None:
                print(f"💡 [LLM建议] 下一个信号相位: {optimal_phase}")
                return optimal_phase
            else:
                print(f"⚠️ [解析失败] 无法从响应中提取相位，保持当前相位 {current_phase}")
                return current_phase
                
        except Exception as e:
            print(f"❌ LLM控制器更新失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return current_phase or 0 