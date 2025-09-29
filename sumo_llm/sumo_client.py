
import os
from sumo_llm.sumo_simulator import SimulationManager

class SumoClient:
    def __init__(self):
        self.manager = None

    def start_simulation(self):
        current_dir = os.getcwd()
        path = "sumo_llm"
        config_file = os.path.join(current_dir, path, "osm.sumocfg")
        junctions_file = os.path.join(current_dir, path, "J54_data.json")
        self.manager = SimulationManager().start_simulation(config_file, junctions_file)
    
    def get_simulation_state(self):
        return self.manager.get_simulation_state()
