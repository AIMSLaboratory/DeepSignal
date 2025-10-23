import os
import time
# import dashscope
# from dashscope import Generation
from .sumo_simulator import SUMOSimulator
from threading import Thread, Event
import queue

class TrafficAnalyzer:
    def __init__(self, api_key: str = 'sk-bc4339fe8e8e48f6b57fab86b5d70afa'):
        self.api_key = api_key
        dashscope.api_key = api_key

    def analyze_traffic(self, vehicle_counts, phase_info):
        prompt = f"""作为交通信号配时专家，请根据以下模板简要分析交通状况：

当前车流量：
{self._format_counts(vehicle_counts)}

当前相位：{phase_info['phase_name']}
剩余时间：{phase_info['remaining_duration']:.1f}秒

请按以下格式回答：
1. 当前状态：[拥堵/正常/空闲]
2. 建议操作：[保持当前相位/切换到XX相位/延长当前相位XX秒]

只需给出简短的状态判断和一条具体建议。"""

        try:
            response = Generation.call(
                model='qwen-plus',
                prompt=prompt,
                seed=1234,
                max_tokens=200,
                temperature=0.1,
                top_p=0.8,
                result_format='message',
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                return f"API调用失败: {response.code}"
                
        except Exception as e:
            return f"分析失败: {str(e)}"

    def _format_counts(self, counts):
        return "\n".join([f"{direction}向: {count}辆" for direction, count in counts.items()])

def run_simulation(simulator, stop_event):
    """持续运行仿真的线程函数"""
    while not stop_event.is_set():
        if not simulator.step():
            break
        time.sleep(1)  # 控制仿真速度

def handle_user_input(simulator, analyzer, test_junction, stop_event, result_queue):
    """处理用户输入的线程函数"""
    print("仿真已启动。输入'yes'进行交通分析，输入'quit'退出")
    
    while not stop_event.is_set():
        try:
            user_input = input().strip().lower()
            
            if user_input == 'quit':
                stop_event.set()
                break
                
            elif user_input == 'yes':
                try:
                    # 获取当前状态
                    counts, _ = simulator.get_junction_vehicle_counts(test_junction)
                    phase_info = simulator.get_current_phase(test_junction)
                    
                    if counts and phase_info:
                        # 将结果放入队列
                        result_queue.put(("data", {
                            "counts": counts,
                            "phase_info": phase_info
                        }))
                        
                        # 进行分析
                        analysis = analyzer.analyze_traffic(counts, phase_info)
                        result_queue.put(("analysis", analysis))
                    else:
                        result_queue.put(("error", "数据获取失败"))
                except Exception as e:
                    result_queue.put(("error", f"分析过程出错: {str(e)}"))
        except EOFError:
            break

def print_results(result_queue, stop_event):
    """打印结果的线程函数"""
    while not stop_event.is_set():
        try:
            msg_type, content = result_queue.get(timeout=1)
            
            if msg_type == "data":
                print("\n当前数据：")
                print(f"车流量: {content['counts']}")
                print(f"相位信息: {content['phase_info']}")
            elif msg_type == "analysis":
                print("\n分析结果：")
                print(content)
                print("\n继续输入'yes'进行新的分析，或输入'quit'退出")
            elif msg_type == "error":
                print(f"\n错误: {content}")
                
        except queue.Empty:
            continue

def main():
    # 设置文件路径
    current_dir = os.getcwd()
    config_file = os.path.join(current_dir, "osm.sumocfg")
    junctions_file = os.path.join(current_dir, "junctions_data.json")
    
    # 初始化
    simulator = SUMOSimulator(config_file, junctions_file, gui=True)
    analyzer = TrafficAnalyzer(api_key="sk-key")

    # 启动仿真
    if not simulator.start_simulation():
        print("仿真启动失败")
        return

    # 获取测试路口ID
    test_junction = '1159176756' # list(simulator.junctions_data.keys())[0]

    # 创建线程同步对象
    stop_event = Event()
    result_queue = queue.Queue()

    try:
        # 创建并启动线程
        sim_thread = Thread(target=run_simulation, args=(simulator, stop_event))
        input_thread = Thread(target=handle_user_input, 
                            args=(simulator, analyzer, test_junction, stop_event, result_queue))
        output_thread = Thread(target=print_results, args=(result_queue, stop_event))
        
        sim_thread.daemon = True
        input_thread.daemon = True
        output_thread.daemon = True
        
        sim_thread.start()
        input_thread.start()
        output_thread.start()
        
        # 等待线程结束
        sim_thread.join()
        input_thread.join()
        output_thread.join()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        stop_event.set()
        simulator.close()
        print("仿真已关闭")

if __name__ == "__main__":
    main()