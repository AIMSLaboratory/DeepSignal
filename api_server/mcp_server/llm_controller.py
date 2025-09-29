#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大语言模型控制器实现
使用LLM进行智能信号控制决策

输入特点：
实时性：需要当前实时队列数据
完整性：需要完整的进出车道信息
连续性：需要持续的相位和时间信息
输出特点：
单一决策：输出单个最优相位ID
实时响应：每次调用都给出新的决策
预测性：基于未来状态预测做出决策
这个预测控制算法的输入输出设计比较合理，能够满足基本的交通控制需求。但如果要进一步提升效果，可以考虑：
输入端改进：
添加车速信息
考虑车辆类型
引入天气数据
加入特殊事件信息
输出端扩展：
输出置信度
提供备选方案
添加预期效果评估
包含风险评估指标
"""

import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

class LLMController:
    def __init__(self, client: Any, context_window: int = 2000):
        """
        初始化LLM控制器
        
        参数:
            client: LLM客户端（支持Claude、Gemma等）
            context_window: 上下文窗口大小
        """
        self.client = client
        self.context_window = context_window
        self.conversation_history = {}
        self.min_phase_duration = 15
        self.max_phase_duration = 90
    
    def _format_context(self, context_id: str,
                       current_phase: int,
                       queue_lengths: Dict[int, int],
                       historical_data: Dict[str, Any]) -> str:
        """
        格式化上下文信息
        
        参数:
            context_id: 上下文ID
            current_phase: 当前相位
            queue_lengths: 当前队列长度
            historical_data: 历史数据
        
        返回:
            格式化的上下文字符串
        """
        context = []
        
        # 添加当前状态
        context.append("当前状态:")
        context.append(f"- 当前相位: {current_phase}")
        context.append("- 当前队列长度:")
        for lane, length in queue_lengths.items():
            context.append(f"  - 车道 {lane}: {length} 辆")
        
        # 添加相位定义
        if 'phases' in historical_data:
            context.append("\n相位定义:")
            for phase_id, movements in historical_data['phases'].items():
                context.append(f"- 相位 {phase_id}: {movements}")
        
        # 添加历史性能指标
        if 'performance_metrics' in historical_data:
            metrics = historical_data['performance_metrics']
            context.append("\n历史性能指标:")
            for metric, value in metrics.items():
                context.append(f"- {metric}: {value}")
        
        # 添加预测数据
        if 'predicted_data' in historical_data:
            pred_data = historical_data['predicted_data']
            context.append("\n预测数据:")
            context.append(f"- 预测时间范围: {pred_data.get('prediction_horizon')} 秒")
            context.append(f"- 预测置信度: {pred_data.get('confidence', 0):.2f}")
            
            if 'predicted_incoming_queues' in pred_data:
                context.append("- 预测进入车道队列:")
                for lane, length in pred_data['predicted_incoming_queues'].items():
                    context.append(f"  - 车道 {lane}: {length} 辆")
        
        return "\n".join(context)
    
    def _generate_prompt(self, context: str) -> str:
        """
        生成提示词
        
        参数:
            context: 上下文信息
        
        返回:
            完整的提示词
        """
        prompt = f"""作为一个智能交通信号控制系统，请基于以下信息为交叉口选择最优的下一个信号相位：

            {context}

            请考虑以下因素：
            1. 当前各车道的排队长度
            2. 历史性能数据
            3. 预测的未来交通状态
            4. 相位切换的平滑性

            请提供：
            1. 建议的下一个相位ID
            2. 建议的相位持续时间（秒）
            3. 决策理由
            4. 预期效果

            请以JSON格式回复，包含以下字段：
            {
                "optimal_phase": int,  // 最优相位ID
                "duration": int,  // 建议持续时间（秒）
                "reasoning": string,  // 决策理由
                "confidence": float,  // 决策置信度（0-1）
                "predicted_impact": string  // 预期影响
            }"""
        return prompt
    
    def optimize_signal_timing(self, context_id: str,
                             intersection_id: int,
                             current_phase: int,
                             queue_lengths: Dict[int, int],
                             historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化信号配时
        
        参数:
            context_id: 对话上下文ID
            intersection_id: 交叉口ID
            current_phase: 当前相位
            queue_lengths: 当前队列长度
            historical_data: 历史数据
        
        返回:
            优化结果字典
        """
        try:
            # 格式化上下文
            context = self._format_context(
                context_id,
                current_phase,
                queue_lengths,
                historical_data
            )
            
            # 生成提示词
            prompt = self._generate_prompt(context)
            
            # 调用LLM
            response = self.client.chat(
                messages=[{
                    "role": "system",
                    "content": "你是一个专业的交通信号控制专家。"
                }, {
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # 解析响应
            try:
                result = json.loads(response['content'])
            except:
                # 如果JSON解析失败，尝试提取数字作为相位ID
                import re
                numbers = re.findall(r'\d+', response['content'])
                if numbers:
                    result = {
                        "optimal_phase": int(numbers[0]),
                        "duration": self.min_phase_duration,
                        "reasoning": "解析失败，使用提取的数字作为相位ID",
                        "confidence": 0.5,
                        "predicted_impact": "未知"
                    }
                else:
                    raise ValueError("无法从响应中提取相位ID")
            
            # 验证和规范化结果
            optimal_phase = int(result.get("optimal_phase", current_phase))
            duration = min(max(int(result.get("duration", self.min_phase_duration)),
                            self.min_phase_duration),
                         self.max_phase_duration)
            
            return {
                "status": "success",
                "optimal_phase": optimal_phase,
                "duration": duration,
                "reasoning": result.get("reasoning", "未提供理由"),
                "confidence": float(result.get("confidence", 0.5)),
                "predicted_impact": result.get("predicted_impact", "未提供预期影响")
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"LLM优化失败: {str(e)}",
                "optimal_phase": current_phase,
                "duration": self.min_phase_duration
            }
    
    def update(self, incoming_queues: Dict[int, int],
              outgoing_queues: Dict[int, int],
              current_phase: Optional[int] = None,
              current_duration: float = 0) -> int:
        """
        更新并返回最优相位（简化接口）
        
        参数:
            incoming_queues: 进入车道队列长度字典
            outgoing_queues: 离开车道队列长度字典
            current_phase: 当前相位ID（如果有）
            current_duration: 当前相位持续时间
        
        返回:
            最优相位ID
        """
        try:
            result = self.optimize_signal_timing(
                context_id=f"default_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                intersection_id=0,
                current_phase=current_phase or 0,
                queue_lengths={**incoming_queues, **outgoing_queues},
                historical_data={
                    "current_phase_duration": current_duration
                }
            )
            
            if result["status"] == "success":
                return result["optimal_phase"]
            else:
                return current_phase or 0
                
        except Exception as e:
            print(f"LLM控制器更新失败: {str(e)}")
            return current_phase or 0 