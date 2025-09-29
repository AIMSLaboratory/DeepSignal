import os
import sys
import traci
import json
from collections import defaultdict
from sumolib import net

class JunctionInfoExtractor:
    def __init__(self, net_file):
        """
        初始化提取器
        net_file: SUMO网络文件路径 (.net.xml)
        """
        self.net_file = net_file
        self.net = net.readNet(net_file)
        
    def get_edge_direction(self, edge):
        """确定边的方向"""
        shape = edge.getShape()
        if len(shape) < 2:
            return "U"  # Unknown
            
        x1, y1 = shape[0]
        x2, y2 = shape[-1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) > abs(dy):
            return "E" if dx > 0 else "W"
        else:
            return "N" if dy > 0 else "S"
    
    def is_traffic_light_controlled(self, junction_id):
        """检查交叉口是否由信号灯控制"""
        try:
            # 获取所有信号灯
            traffic_lights = self.net.getTrafficLights()
            # 检查交叉口ID是否在任何信号灯的控制节点中
            for tls in traffic_lights:
                if junction_id in str(tls):  # 简单的字符串匹配
                    return True, tls.getID()
            return False, None
        except:
            return False, None
            
    def get_junction_info(self):
        """获取所有交叉口和相关车道信息"""
        junctions_data = {}
        
        # 获取所有交叉口
        for junction in self.net.getNodes():
            junction_id = junction.getID()
            
            # 跳过内部交叉口
            if junction_id.startswith(":"):
                continue
                
            # 获取交叉口位置
            x, y = junction.getCoord()
            
            # 获取进入该交叉口的边
            incoming_edges = junction.getIncoming()
            
            # 获取离开该交叉口的边
            outgoing_edges = junction.getOutgoing()
            
            # 按方向组织进入车道信息
            incoming_lanes_by_direction = defaultdict(list)
            
            for edge in incoming_edges:
                edge_id = edge.getID()
                direction = self.get_edge_direction(edge)
                
                # 获取该边的所有车道
                for lane in edge.getLanes():
                    lane_id = lane.getID()
                    lane_index = lane.getIndex()
                    
                    lane_info = {
                        "lane_id": lane_id,
                        "lane_index": lane_index,
                        "edge_id": edge_id,
                        "direction": direction,
                        "length": lane.getLength(),
                        "speed_limit": lane.getSpeed()
                    }
                    incoming_lanes_by_direction[direction].append(lane_info)
            
            # 按方向组织离开车道信息
            outgoing_lanes_by_direction = defaultdict(list)
            
            for edge in outgoing_edges:
                edge_id = edge.getID()
                direction = self.get_edge_direction(edge)
                
                # 获取该边的所有车道
                for lane in edge.getLanes():
                    lane_id = lane.getID()
                    lane_index = lane.getIndex()
                    
                    lane_info = {
                        "lane_id": lane_id,
                        "lane_index": lane_index,
                        "edge_id": edge_id,
                        "direction": direction,
                        "length": lane.getLength(),
                        "speed_limit": lane.getSpeed()
                    }
                    outgoing_lanes_by_direction[direction].append(lane_info)
            
            # 构建交叉口数据
            junction_info = {
                "junction_name": f"Junction_{junction_id}",
                "location": {"x": x, "y": y},
                "incoming_lanes": {
                    direction: sorted(lanes, key=lambda x: (x['edge_id'], x['lane_index']))
                    for direction, lanes in incoming_lanes_by_direction.items()
                    if lanes  # 只包含有车道的方向
                },
                "outgoing_lanes": {
                    direction: sorted(lanes, key=lambda x: (x['edge_id'], x['lane_index']))
                    for direction, lanes in outgoing_lanes_by_direction.items()
                    if lanes  # 只包含有车道的方向
                }
            }
            
            # 添加交叉口类型
            junction_type = junction.getType()
            if junction_type:
                junction_info["type"] = junction_type

            # 检查是否是信号灯控制的交叉口
            is_tls, tls_id = self.is_traffic_light_controlled(junction_id)
            if is_tls:
                junction_info["traffic_light_id"] = tls_id
            
            # 如果该交叉口有进入车道或离开车道，则添加到数据中
            if junction_info["incoming_lanes"] or junction_info["outgoing_lanes"]:
                junctions_data[junction_id] = junction_info
        
        return junctions_data
    
    def export_to_json(self, output_file):
        """导出数据到JSON文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            junctions_data = self.get_junction_info()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(junctions_data, f, indent=2, ensure_ascii=False)
                
            print(f"Successfully exported junction data to {output_file}")
            return junctions_data
            
        except Exception as e:
            print(f"Error exporting junction data: {str(e)}")
            return None

def analyze_junction_data(junctions_data):
    """分析并打印交叉口数据的基本统计信息"""
    if not junctions_data:
        print("No data to analyze")
        return
        
    total_junctions = len(junctions_data)
    total_incoming_lanes = sum(
        sum(len(lanes) for lanes in junction["incoming_lanes"].values())
        for junction in junctions_data.values()
    )
    total_outgoing_lanes = sum(
        sum(len(lanes) for lanes in junction["outgoing_lanes"].values())
        for junction in junctions_data.values()
    )
    
    print(f"\n=== 交叉口数据分析 ===")
    print(f"总交叉口数量: {total_junctions}")
    print(f"总进车道数量: {total_incoming_lanes}")
    print(f"总出车道数量: {total_outgoing_lanes}")
    print("\n方向分布:")
    
    incoming_direction_counts = defaultdict(int)
    outgoing_direction_counts = defaultdict(int)
    
    for junction in junctions_data.values():
        for direction in junction["incoming_lanes"].keys():
            incoming_direction_counts[direction] += 1
        for direction in junction["outgoing_lanes"].keys():
            outgoing_direction_counts[direction] += 1
    
    print("进入车道方向分布:")
    for direction, count in incoming_direction_counts.items():
        print(f"{direction}: {count}个交叉口有该方向的进车道")
    
    print("\n离开车道方向分布:")
    for direction, count in outgoing_direction_counts.items():
        print(f"{direction}: {count}个交叉口有该方向的出车道")
    
    print("\n每个交叉口的车道详情:")
    for junction_id, junction in junctions_data.items():
        print(f"\n交叉口 {junction_id}:")
        if "traffic_light_id" in junction:
            print(f"  信号灯ID: {junction['traffic_light_id']}")
            
        print("  进入车道:")
        for direction, lanes in junction["incoming_lanes"].items():
            print(f"    {direction}方向: {len(lanes)}条车道")
            for lane in lanes:
                print(f"      - 车道ID: {lane['lane_id']}, 长度: {lane['length']:.1f}m, 限速: {lane['speed_limit']*3.6:.1f}km/h")
        
        print("  离开车道:")
        for direction, lanes in junction["outgoing_lanes"].items():
            print(f"    {direction}方向: {len(lanes)}条车道")
            for lane in lanes:
                print(f"      - 车道ID: {lane['lane_id']}, 长度: {lane['length']:.1f}m, 限速: {lane['speed_limit']*3.6:.1f}km/h")

def main():
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 请替换为你的网络文件路径
    net_file = os.path.join(current_dir, "ChengduCity.net.xml")
    output_file = os.path.join(current_dir, "J54_data2.json")
    
    if not os.path.exists(net_file):
        print(f"Error: Network file not found at {net_file}")
        print("Please provide the correct path to your .net.xml file")
        return
    
    try:
        extractor = JunctionInfoExtractor(net_file)
        junctions_data = extractor.export_to_json(output_file)
        
        if junctions_data:
            analyze_junction_data(junctions_data)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure you have the correct SUMO installation and the network file is valid")

if __name__ == "__main__":
    main()