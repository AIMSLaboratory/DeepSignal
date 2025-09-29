#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例代码：展示如何使用各种交通信号控制算法
"""

from typing import Dict, List, Tuple
from max_pressure import MaxPressureAlgorithm
from prediction_optimizer import PredictionOptimizer
from llm_controller import LLMController
from datetime import datetime

def create_test_intersection() -> Dict[int, List[Tuple[int, int]]]:
    """创建测试交叉口的相位定义"""
    phases = {
        0: [(101, 203), (103, 201)],  # 南北方向直行
        1: [(102, 204), (104, 202)],  # 东西方向直行
        2: [(101, 202), (103, 204)],  # 南北方向左转
        3: [(102, 201), (104, 203)]   # 东西方向左转
    }
    return phases

def create_test_queues() -> Tuple[Dict[int, int], Dict[int, int]]:
    """创建测试队列数据"""
    # 进入车道队列
    incoming_queues = {
        101: 10,  # 南进口
        102: 15,  # 东进口
        103: 8,   # 北进口
        104: 12   # 西进口
    }
    
    # 离开车道队列
    outgoing_queues = {
        201: 0,   # 北出口
        202: 0,   # 东出口
        203: 0,   # 南出口
        204: 0    # 西出口
    }
     
    return incoming_queues, outgoing_queues

def test_max_pressure():
    """测试Max Pressure算法"""
    print("\n=== 测试 Max Pressure 算法 ===")
    
    # 创建算法实例
    phases = create_test_intersection()
    controller = MaxPressureAlgorithm(phases)
    
    # 创建测试数据
    incoming_queues, outgoing_queues = create_test_queues()
    
    # 获取最优相位
    optimal_phase = controller.update(
        incoming_queues=incoming_queues,
        outgoing_queues=outgoing_queues,
        current_phase=0,
        current_duration=20
    )
    
    print(f"当前相位: 0")
    print(f"进入车道队列: {incoming_queues}")
    print(f"离开车道队列: {outgoing_queues}")
    print(f"最优相位: {optimal_phase}")

def test_prediction():
    """测试预测优化算法"""
    print("\n=== 测试预测优化算法 ===")
    
    # 创建算法实例
    phases = create_test_intersection()
    controller = PredictionOptimizer(phases)
    
    # 创建测试数据
    incoming_queues, outgoing_queues = create_test_queues()
    
    # 更新历史数据
    controller.update_historical_data(
        timestamp=datetime.now(),
        incoming_queues=incoming_queues,
        outgoing_queues=outgoing_queues,
        active_phase=0
    )
    
    # 获取最优相位
    optimal_phase = controller.update(
        incoming_queues=incoming_queues,
        outgoing_queues=outgoing_queues,
        current_phase=0,
        current_duration=20
    )
    
    print(f"当前相位: 0")
    print(f"进入车道队列: {incoming_queues}")
    print(f"离开车道队列: {outgoing_queues}")
    print(f"最优相位: {optimal_phase}")

def test_llm_controller():
    """测试LLM控制器"""
    print("\n=== 测试 LLM 控制器 ===")
    
    try:
        # 创建LLM客户端（这里使用模拟客户端）
        class MockLLMClient:
            def chat(self, messages):
                return {
                    "content": '''
                    {
                        "optimal_phase": 2,
                        "duration": 45,
                        "reasoning": "南北方向左转车道排队较长",
                        "confidence": 0.85,
                        "predicted_impact": "预计可减少南北方向左转车道排队长度30%"
                    }
                    '''
                }
        
        # 创建算法实例
        controller = LLMController(client=MockLLMClient())
        
        # 创建测试数据
        incoming_queues, outgoing_queues = create_test_queues()
        
        # 获取最优相位
        result = controller.optimize_signal_timing(
            context_id="test_1",
            intersection_id=1,
            current_phase=0,
            queue_lengths={**incoming_queues, **outgoing_queues},
            historical_data={
                "phases": create_test_intersection(),
                "current_phase_duration": 20
            }
        )
        
        print(f"当前相位: 0")
        print(f"进入车道队列: {incoming_queues}")
        print(f"离开车道队列: {outgoing_queues}")
        print(f"优化结果:")
        print(f"- 状态: {result['status']}")
        print(f"- 最优相位: {result['optimal_phase']}")
        print(f"- 建议持续时间: {result['duration']}秒")
        print(f"- 决策理由: {result['reasoning']}")
        print(f"- 置信度: {result['confidence']}")
        print(f"- 预期影响: {result['predicted_impact']}")
        
    except Exception as e:
        print(f"测试LLM控制器时出错: {str(e)}")

def main():
    """主函数"""
    # 测试各个算法
    test_max_pressure()
    test_prediction()
    test_llm_controller()

if __name__ == "__main__":
    main() 