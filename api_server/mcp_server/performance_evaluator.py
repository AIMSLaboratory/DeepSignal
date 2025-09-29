#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交通信号控制性能评估器
用于评估不同交通控制策略的效果，收集性能指标并生成优化建议。
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any, Optional, List
from sumo_llm.sumo_simulator import get_simulator

class PerformanceEvaluator:
    def __init__(self, config_path=None):
        """
        初始化性能评估器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.metrics = {
            "waiting_time": [],         # 等待时间
            "queue_length": [],         # 队列长度
            "throughput": [],           # 通行量
            "halting_count": [],        # 停止的车辆数
            "mean_speed": []            # 平均速度
        }
        
        self.algorithm_performance = {
            "max_pressure": {"score": 0, "evaluations": 0},
            "prediction_optimizer": {"score": 0, "evaluations": 0},
            "llm_controller": {"score": 0, "evaluations": 0}
        }
        
        self.history = []
        self.config = self._load_config(config_path)
        self.last_collection_time = time.time()
        self.collection_interval = self.config.get("collection_interval", 60)  # 默认60秒收集一次
        self.current_algorithm = "max_pressure"  # 记录当前使用的算法
        
    def _load_config(self, config_path):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        # 加载配置文件，如果不存在则使用默认配置
        default_config = {
            "collection_interval": 60,  # 秒
            "evaluation_window": 3600,  # 1小时的评估窗口
            "weights": {
                "waiting_time": 0.3,    # 等待时间权重
                "queue_length": 0.3,    # 队列长度权重
                "throughput": 0.2,      # 通行量权重
                "mean_speed": 0.1,      # 平均速度权重
                "halting_count": 0.1    # 停止车辆数权重
            },
            "thresholds": {
                "waiting_time": 120,    # 秒
                "queue_length": 15,     # 车辆数
                "mean_speed": 5         # m/s
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                return default_config
        return default_config
    
    def set_current_algorithm(self, algorithm_name: str):
        """
        设置当前使用的算法
        
        Args:
            algorithm_name: 算法名称
        """
        if algorithm_name in self.algorithm_performance:
            self.current_algorithm = algorithm_name
    
    def collect_metrics(self, tl_id: str) -> Dict[str, Any]:
        """
        收集当前交通状况的各项指标
        
        Args:
            tl_id: 交通信号灯ID
            
        Returns:
            收集的指标数据
        """
        simulator = get_simulator()
        if simulator is None:
            return {"status": "error", "message": "SUMO模拟器未初始化"}
        
        current_time = time.time()
        if current_time - self.last_collection_time < self.collection_interval:
            return {"status": "skipped", "message": "收集间隔未到"}
        
        self.last_collection_time = current_time
        
        # 收集指标
        metrics = {}
        
        try:
            # 获取交通数据
            traffic_data, _ = simulator.get_junction_vehicle_counts(tl_id)
            
            if not traffic_data:
                return {"status": "error", "message": "无法获取交通数据"}
            
            # 计算总体指标
            total_vehicles = 0
            total_waiting_time = 0
            total_halting = 0
            total_speed = 0
            
            for direction, data in traffic_data.items():
                total_vehicles += data['vehicle_count']
                total_waiting_time += data['waiting_time'] * data['vehicle_count'] if data['vehicle_count'] > 0 else 0
                total_halting += data['halting_count']
                total_speed += data['mean_speed'] * data['vehicle_count'] if data['vehicle_count'] > 0 else 0
            
            # 平均等待时间
            metrics["waiting_time"] = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
            
            # 获取队列长度 (通过phases pressure计算)
            phase_queues = simulator.calculate_all_phases_pressure(tl_id)
            total_queue = sum([p["incoming_queue_length"] for p in phase_queues.values()])
            metrics["queue_length"] = total_queue
            
            # 通行量
            metrics["throughput"] = total_vehicles
            
            # 停止的车辆数
            metrics["halting_count"] = total_halting
            
            # 平均速度
            metrics["mean_speed"] = total_speed / total_vehicles if total_vehicles > 0 else 0
            
            # 记录指标
            for key, value in metrics.items():
                self.metrics[key].append(value)
                # 保持数据量在合理范围内
                if len(self.metrics[key]) > 1000:
                    self.metrics[key] = self.metrics[key][-1000:]
            
            # 记录当前算法和相位
            current_phase = simulator.get_current_phase(tl_id)
            
            self.history.append({
                "timestamp": current_time,
                "metrics": metrics,
                "phase": current_phase,
                "algorithm": self.current_algorithm
            })
            
            return {"status": "success", "metrics": metrics}
            
        except Exception as e:
            return {"status": "error", "message": f"收集指标失败: {str(e)}"}
    
    def evaluate_algorithm(self, algorithm_name: str, metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        评估指定算法的性能
        
        Args:
            algorithm_name: 算法名称
            metrics: 指标数据，如果为None则使用最近收集的指标
            
        Returns:
            评估结果
        """
        if metrics is None:
            # 使用最近收集的指标
            if not self.metrics["waiting_time"]:
                return {"status": "error", "message": "没有可用的指标数据"}
            metrics = {k: self.metrics[k][-1] for k in self.metrics.keys()}
        
        # 计算性能得分
        score = 0
        weights = self.config["weights"]
        thresholds = self.config["thresholds"]
        
        # 等待时间得分（越低越好）
        waiting_time_score = max(0, 1 - metrics["waiting_time"] / thresholds["waiting_time"])
        score += waiting_time_score * weights["waiting_time"]
        
        # 队列长度得分（越短越好）
        queue_length_score = max(0, 1 - metrics["queue_length"] / thresholds["queue_length"])
        score += queue_length_score * weights["queue_length"]
        
        # 通行量得分（越高越好）
        throughput_score = min(1, metrics["throughput"] / 50)  # 假设最大通行量为50
        score += throughput_score * weights["throughput"]
        
        # 停止车辆数得分（越少越好）
        halting_score = max(0, 1 - metrics["halting_count"] / metrics["throughput"] if metrics["throughput"] > 0 else 0)
        score += halting_score * weights["halting_count"]
        
        # 平均速度得分（越高越好）
        speed_score = min(1, metrics["mean_speed"] / thresholds["mean_speed"])
        score += speed_score * weights["mean_speed"]
        
        # 更新算法性能记录
        if algorithm_name in self.algorithm_performance:
            current_score = self.algorithm_performance[algorithm_name]["score"]
            evaluations = self.algorithm_performance[algorithm_name]["evaluations"]
            
            # 使用移动平均更新得分
            new_score = (current_score * evaluations + score) / (evaluations + 1)
            self.algorithm_performance[algorithm_name]["score"] = new_score
            self.algorithm_performance[algorithm_name]["evaluations"] += 1
        
        return {
            "status": "success",
            "algorithm": algorithm_name,
            "score": score,
            "detailed_scores": {
                "waiting_time": waiting_time_score,
                "queue_length": queue_length_score,
                "throughput": throughput_score,
                "halting_count": halting_score,
                "mean_speed": speed_score
            }
        }
    
    def get_optimization_suggestion(self, tl_id: str) -> Dict[str, Any]:
        """
        基于历史性能生成优化建议
        
        Args:
            tl_id: 交通信号灯ID
            
        Returns:
            优化建议
        """
        # 找到性能最好的算法
        best_algorithm = max(self.algorithm_performance.items(), 
                            key=lambda x: x[1]["score"] if x[1]["evaluations"] > 0 else -1)
        
        # 分析当前交通状况
        simulator = get_simulator()
        phase_queues = simulator.calculate_all_phases_pressure(tl_id)
        current_phase = simulator.get_current_phase(tl_id)
        
        # 检测特殊情况
        is_congested = any([p["incoming_queue_length"] > self.config["thresholds"]["queue_length"] 
                           for p in phase_queues.values()])
        
        # 生成建议
        suggestions = []
        
        if best_algorithm[0] != "none" and best_algorithm[1]["evaluations"] > 0:
            suggestions.append(f"建议使用 {best_algorithm[0]} 算法，其历史性能得分为 {best_algorithm[1]['score']:.2f}")
        
        if is_congested:
            suggestions.append("当前交通拥堵，建议延长绿灯时间或采用自适应控制")
        
        # 分析趋势
        if len(self.metrics["waiting_time"]) > 10:
            recent_waiting = np.mean(self.metrics["waiting_time"][-10:])
            older_waiting = np.mean(self.metrics["waiting_time"][-20:-10]) if len(self.metrics["waiting_time"]) >= 20 else recent_waiting
            
            if recent_waiting > older_waiting * 1.2:
                suggestions.append("等待时间呈上升趋势，建议调整控制策略")
            elif recent_waiting < older_waiting * 0.8:
                suggestions.append("等待时间呈下降趋势，当前控制策略有效")
        
        return {
            "status": "success",
            "best_algorithm": best_algorithm[0] if best_algorithm[1]["evaluations"] > 0 else None,
            "suggestions": suggestions,
            "algorithm_performance": self.algorithm_performance
        }
    
    def auto_optimize(self, tl_id: str, max_pressure_obj=None, prediction_optimizer_obj=None, llm_controller_obj=None) -> Dict[str, Any]:
        """
        自动根据性能评估结果调整控制策略
        
        Args:
            tl_id: 交通信号灯ID
            max_pressure_obj: MaxPressure算法对象
            prediction_optimizer_obj: 预测优化算法对象
            llm_controller_obj: LLM控制器对象
            
        Returns:
            优化结果
        """
        suggestion = self.get_optimization_suggestion(tl_id)
        
        if suggestion["status"] != "success" or not suggestion["best_algorithm"]:
            return {"status": "error", "message": "无法获取优化建议"}
        
        # 切换到最佳算法
        best_algorithm = suggestion["best_algorithm"]
        
        # 获取当前交通状况
        simulator = get_simulator()
        phase_queues = simulator.get_phase_queues_from_sumo(tl_id)
        current_phase = simulator.get_current_phase(tl_id)
        
        # 根据最佳算法获取最优相位
        result = None
        if best_algorithm == "max_pressure" and max_pressure_obj:
            result = max_pressure_obj.update(phase_queues, current_phase["phase_index"], current_phase["remaining_duration"])
        elif best_algorithm == "prediction_optimizer" and prediction_optimizer_obj:
            result = prediction_optimizer_obj.update(phase_queues, current_phase["phase_index"], current_phase["remaining_duration"])
        elif best_algorithm == "llm_controller" and llm_controller_obj:
            result = llm_controller_obj.update(phase_queues, current_phase["phase_index"], current_phase["remaining_duration"])
        else:
            return {"status": "error", "message": f"不支持的算法或算法对象未提供: {best_algorithm}"}
        
        # 应用最优相位
        simulator.set_phase_switch(tl_id, result)
        
        # 更新当前算法
        self.set_current_algorithm(best_algorithm)
        
        return {
            "status": "success",
            "applied_algorithm": best_algorithm,
            "optimal_phase": result,
            "message": "已自动切换到最佳控制策略"
        }
    
    def generate_report(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        生成性能评估报告
        
        Args:
            time_window: 时间窗口（秒），None表示使用全部历史数据
            
        Returns:
            评估报告
        """
        if not self.history:
            return {"status": "error", "message": "没有历史数据"}
        
        # 默认使用全部历史数据
        report_data = self.history
        
        # 如果指定了时间窗口，筛选数据
        if time_window:
            current_time = time.time()
            report_data = [item for item in self.history 
                          if current_time - item["timestamp"] <= time_window]
        
        if not report_data:
            return {"status": "error", "message": "指定时间窗口内没有数据"}
        
        # 计算平均指标
        avg_metrics = {metric: np.mean([item["metrics"][metric] for item in report_data if metric in item["metrics"]]) 
                      for metric in self.metrics.keys()}
        
        # 计算算法性能统计
        algorithm_stats = {}
        for item in report_data:
            alg = item["algorithm"]
            if alg not in algorithm_stats:
                algorithm_stats[alg] = {
                    "count": 0, 
                    "waiting_time": [], 
                    "queue_length": [],
                    "throughput": [],
                    "halting_count": [],
                    "mean_speed": []
                }
            
            algorithm_stats[alg]["count"] += 1
            for metric in self.metrics.keys():
                if metric in item["metrics"]:
                    algorithm_stats[alg][metric].append(item["metrics"][metric])
        
        # 计算每个算法的平均指标
        for alg in algorithm_stats:
            for metric in self.metrics.keys():
                if algorithm_stats[alg][metric]:
                    algorithm_stats[alg][f"avg_{metric}"] = np.mean(algorithm_stats[alg][metric])
                else:
                    algorithm_stats[alg][f"avg_{metric}"] = 0
        
        return {
            "status": "success",
            "report_time": time.time(),
            "time_window": time_window,
            "data_points": len(report_data),
            "average_metrics": avg_metrics,
            "algorithm_statistics": algorithm_stats,
            "algorithm_performance": self.algorithm_performance
        } 