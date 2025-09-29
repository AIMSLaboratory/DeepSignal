import xml.etree.ElementTree as ET
import random

def process_routes(input_file, output_file, keep_ratio=0.5, edge_filter="37132266#4", edge_keep_ratio=0.7):
    # 解析XML文件
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # 获取所有车辆元素
    vehicles = root.findall('vehicle')
    total_vehicles = len(vehicles)
    
    # 第一轮：随机选择要保留的车辆
    keep_num = int(total_vehicles * keep_ratio)
    kept_vehicles = random.sample(vehicles, keep_num)
    
    # 移除未被选中的车辆
    for vehicle in vehicles:
        if vehicle not in kept_vehicles:
            root.remove(vehicle)
    
    # 第二轮：处理包含特定edge的车辆
    edge_vehicles = []  # 包含特定edge的车辆
    non_edge_vehicles = []  # 不包含特定edge的车辆
    
    for vehicle in kept_vehicles:
        route = vehicle.find('route')
        edges = route.get('edges').split()
        if edge_filter in edges:
            edge_vehicles.append(vehicle)
        else:
            non_edge_vehicles.append(vehicle)
    
    # 对包含特定edge的车辆进行二次筛选
    edge_keep_num = int(len(edge_vehicles) * edge_keep_ratio)
    kept_edge_vehicles = random.sample(edge_vehicles, edge_keep_num)
    
    # 移除未被选中的包含特定edge的车辆
    for vehicle in edge_vehicles:
        if vehicle not in kept_edge_vehicles:
            root.remove(vehicle)
    
    # 保存到新文件
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    return {
        "原始车辆总数": total_vehicles,
        "第一轮筛选后车辆数": keep_num,
        "包含指定edge的车辆数": len(edge_vehicles),
        "最终保留的包含指定edge的车辆数": edge_keep_num,
        "不包含指定edge的车辆数": len(non_edge_vehicles),
        "最终总车辆数": edge_keep_num + len(non_edge_vehicles)
    }

# 使用示例
input_file = 'Chengdu_ori.rou.xml'  # 输入文件名
output_file = 'Chengdu.rou.xml'  # 输出文件名
keep_ratio = 0.3  # 第一轮保留比例
edge_filter = "37132266#4"  # 需要二次筛选的edge
edge_keep_ratio = 0.7  # 包含特定edge的车辆保留比例

# 处理文件并打印统计信息
stats = process_routes(input_file, output_file, keep_ratio, edge_filter, edge_keep_ratio)

# 打印统计信息
for key, value in stats.items():
    print(f"{key}: {value}")