import os

class SumoClient:
    def __init__(self):
        self.simulator = None

    def start_simulation(self):
        current_dir = os.getcwd()
        path = "sumo_llm"
        config_file = os.path.join(current_dir, path, "osm.sumocfg")
        junctions_file = os.path.join(current_dir, path, "J54_data.json")
        
        from .sumo_simulator import SUMOSimulator
        self.simulator = SUMOSimulator(config_file, junctions_file, gui=True)
        return self.simulator.start_simulation()
    
    def get_simulation_state(self):
        if self.simulator and self.simulator.is_connected():
            return {
                "connected": True,
                "simulation_time": self.simulator.get_simulation_time()
            }
        return {"connected": False}
    
    def stop_simulation(self):
        if self.simulator:
            self.simulator.close()
            self.simulator = None
