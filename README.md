# DeepSignal (SUMO + MCP) — Traffic Signal Control via LLM

[中文 README](README_zh.md)

DeepSignal is our fine-tuned large language model for **traffic-signal control**. The current release is **DeepSignal-4B-V1**.

- **Model (Hugging Face)**: [`AIMS2025/DeepSignal`](https://huggingface.co/AIMS2025/DeepSignal)


This repository also contains a SUMO-based simulation stack and an MCP server to run closed-loop interaction between the LLM and traffic simulations, to evaluate the performance of various baseline signal control models/algorithms, and compare with DeepSignal-4B-V1. Currently, this repository does not include the code for fine-tuning the large language model.

## Team

- **Team name**: AIMSLab
- **Team members**: Feng Xiao, Da Lei, Yuzhan Liu, Jinyou Chi, Yabang, Wang.
- **Team leader**: Feng Xiao (homepage: <https://bs.scu.edu.cn/guanlikexue/202403/9185.html>)
- **Contact**: <trains.ai.lab@gmail.com>


## Key idea: Offline + Online training

We fine-tune DeepSignal using a two-stage learning pipeline:

- **Offline learning (SFT)**: supervised fine-tuning on instruction-style data to learn traffic-state analysis and signal-control decision formatting.
- **Online learning (RL with SUMO)**: reinforcement learning by interacting with SUMO simulations (closed-loop), using diverse scenarios under `scenarios/`.

## Scenarios (training vs hold-out evaluation)

During online interaction, we use the SUMO scenarios under `scenarios/`. We also evaluate generalization on **hold-out scenarios** that are **NOT used in training**.

| City/Region | Scenario directory | Config | Usage | Notes |
|---|---|---|---|---|
| Bad Hersfeld | `BadHersfeld_osm_duarouter` | `duarouter.sumocfg` | Train | OSM + duarouter |
| Bad Hersfeld | `BadHersfeld_osm_osm` | `osm.sumocfg` | Train | OSM |
| Bad Hersfeld | `BadHersfeld_prt_src_prt` | `prt.sumocfg` | Train | PRT |
| Bologna | `bologna_acosta_persontrips_run` | `run.sumocfg` | Train | Acosta (persontrips) |
| Bologna | `bologna_acosta_run` | `run.sumocfg` | Train | Acosta |
| Bologna | `bologna_joined_run` | `run.sumocfg` | Train | Joined |
| Bologna | `bologna_pasubio_run` | `run.sumocfg` | Train | Pasubio |
| Doerpfeldstr | `Doerpfeldstr_all_modes` | `all_modes.sumocfg` | Train | Multi-mode |
| Doerpfeldstr | `Doerpfeldstr_output` | `output.sumocfg` | Train | Output config |
| Doerpfeldstr | `Doerpfeldstr_output_flows` | `output_flows.sumocfg` | Train | Flows |
| Doerpfeldstr | `Doerpfeldstr_output_neu` | `output_neu.sumocfg` | Train | Output (neu) |
| Germany motorways | `germany-motorways_run` | `run.sumocfg` | Train | Motorways |
| PORT tutorial | `port_tutorials_port_brunswick_osm` | `osm.sumocfg` | Train | Brunswick OSM |
| PORT tutorial | `port_tutorials_port_l_beck_port_tutorial` | `port_tutorial.sumocfg` | Train | Lübeck tutorial |
| Wildau | `Wildau_flow1_Spaet` | `flow1_Spaet.sumocfg` | Train | Flow config |
| Cologne | `cologne1` | `cologne1.sumocfg` | Eval (hold-out) | Not used in training |
| Cologne | `cologne3` | `cologne3.sumocfg` | Eval (hold-out) | Not used in training |
| Cologne | `cologne8` | `cologne8.sumocfg` | Eval (hold-out) | Not used in training |
| Ingolstadt | `ingolstadt1` | `ingolstadt1.sumocfg` | Eval (hold-out) | Not used in training |
| Ingolstadt | `ingolstadt21` | `ingolstadt21.sumocfg` | Eval (hold-out) | Not used in training |
| Ingolstadt | `ingolstadt7` | `ingolstadt7.sumocfg` | Eval (hold-out) | Not used in training |
| Chengdu | `sumo_llm` | `osm.sumocfg` | Eval (hold-out) | Not used in training |

## Results from SUMO Simulation

### Evaluation metrics

We evaluate DeepSignal in **SUMO simulation** using intersection-level metrics computed from the simulator:

- **Avg Saturation** (`average_saturation`)
- **Avg Queue Length** (`average_queue_length`)
- **Max Saturation** (`max_saturation`)
- **Max Queue Length** (`max_queue_length`)
- **Congestion Index (0–1)** (`congestion_index`)
- **Congestion Level** (`congestion_level`) and its distribution (%)

#### Metric computation (formulas)

Let $t$ index simulation steps in a time window, and $l$ index controlled lanes at an intersection. In our implementation we use `lane_length = 100m` and `avg_vehicle_length = 5m`.

- Per-lane saturation: $s_{t,l} = \dfrac{(n_{t,l} + h_{t,l})\cdot 5}{100}$, where $n_{t,l}$ is the number of vehicles on lane $l$ at step $t$, and $h_{t,l}$ is the number of halting vehicles.
- Per-lane queue length (meters): $q_{t,l} = h_{t,l}\cdot 5$
- Step averages over valid lanes $L_t$:
```math
\bar{s}_t=\frac{1}{L_t} \sum_{l=1}^{L_t} s_{t,l}, \quad \bar{q}_t=\frac{1}{L_t} \sum_{l=1}^{L_t} q_{t,l}
```
- Window metrics over $T$ steps:
  - `average_saturation` $= \dfrac{1}{T}\sum_{t=1}^{T}\bar{s}_t$
  - `average_queue_length` $= \dfrac{1}{T}\sum_{t=1}^{T}\bar{q}_t$
  - `max_saturation` $= \max_t \bar{s}_t$
  - `max_queue_length` $= \max_t \bar{q}_t$

### Performance Metrics Comparison by Model

| Model | Avg Saturation | Avg Queue Length | Max Saturation | Max Queue Length | Avg Congestion Index |
|---|---:|---:|---:|---:|---:|
| [`Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | 0.1550 | 5.5000 | 0.1550 | 5.4995 | 0.1500 |
| **DeepSignal-4B (Ours)** | 0.1580 | 5.5500 | 0.1580 | 5.5498 | 0.1550 |
| [`LightGPT-8B-Llama3`](https://huggingface.co/lightgpt/LightGPT-8B-Llama3) | 0.1720 | 6.1000 | 0.1720 | 6.1000 | 0.1950 |
| SFT | 0.1780 | 6.2500 | 0.1780 | 6.2500 | 0.2050 |
| Last Round GRPO | 0.1850 | 6.4500 | 0.1850 | 6.4500 | 0.2150 |
| [`Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 0.1980 | 7.2000 | 0.1980 | 7.1989 | 0.2450 |
| Max Pressure | 0.2050 | 7.8000 | 0.2049 | 7.7968 | 0.2550 |
| [`GPT-OSS-20B`](https://huggingface.co/openai/gpt-oss-20b) | 0.2250 | 8.5001 | 0.2250 | 8.4933 | 0.3050 |


## Chengdu Real-world Deployment Comparison

This section reports a **real-world deployment** comparison between LLM-based signal control (marked as `Current` in the figure) and a baseline strategy (Fixed signal timing plan optimized by local traffic management department, marked as `Yesterday` in the figure) in the same intersection, on different days (2025-12-25 and 2025-12-24) during the same time period (14:10:05-17:17:00). The visualization digits come from identified data of the CCTV traffic camera footage.

### Metric computation (real-world)

The congestion index is constructed hierarchically from **phase → intersection → minute → cumulative** time scales:

1) **Phase-level congestion score**  
Assume an intersection has $P$ signal phases. Let $q_p(t)$ be the observed vehicle count (or discharged vehicles) for phase $p$ during a unit time at sampling time $t$. The empirical phase capacity $C_p$ is estimated from historical observations:

$$
C_p = \max_{t \in \mathcal{T}_{\text{hist}}} q_p(t)
$$

The instantaneous phase congestion score is:

$$
s_p(t) = \min \left( 100 \cdot \frac{q_p(t)}{C_p}, 100 \right)
$$

2) **Intersection-level instantaneous score**

$$
S(t) = \frac{1}{P} \sum_{p=1}^{P} s_p(t)
$$

3) **Minute-level congestion index** (multiple samples per minute)  
Let minute $m$ contain $N_m$ valid samples $t_1,\dots,t_{N_m}$:

$$
\bar{S}(m) =
\begin{cases}
\frac{1}{N_m} \sum_{i=1}^{N_m} S(t_i), & N_m > 0, \\
0, & N_m = 0.
\end{cases}
$$

4) **Cumulative congestion index** (from the start reference to minute $T$)

$$
CI(T) = \sum_{m=1}^{T} \bar{S}(m)
$$

### Visual comparison
Congestion index time-series comparison:
![Congestion Index Time-series Comparison](images/congestion_index_timeseries_comparison.gif)

Cumulative congestion index comparison:
![Cumulative Congestion Index Comparison](images/congestion_index_cumulative_comparison.png)


## Model files (GGUF) and local inference

If you are looking for GGUF files for local inference (`llama.cpp` / LM Studio), check the model card in Hugging Face and the packaging notes under `hf/`.

Example (llama.cpp):

```bash
llama-cli -m DeepSignal-4B_V1.F16.gguf -p "You are a traffic management expert. You can use your traffic knowledge to solve the traffic signal control task.
Based on the given traffic scene and state, predict the next signal phase and its duration.
You must answer directly, the format must be: next signal phase: {number}, duration: {seconds} seconds
where the number is the phase index (starting from 0) and the seconds is the duration (usually between 20-90 seconds)."
```

## Environment setup

### SUMO installation

1. Install SUMO from the official website: <https://sumo.dlr.de/docs/Downloads.php>
2. Set `SUMO_HOME` (examples):
   - Linux/Mac: `export SUMO_HOME="/usr/local/share/sumo"`
   - Mac (example): `export SUMO_HOME="/Users/<you>/sumo/bin"`

### Python dependencies (uv)

```bash
pip install uv
uv venv
source .venv/bin/activate
uv sync
```

## Evaluation workflow (reproducible)

The evaluation is performed in SUMO simulation and recorded as CSV metrics under `results/`.

### Start the MCP server + SUMO simulation

```bash
source .venv/bin/activate
export SUMO_HOME="/Users/<you>/sumo/bin"
uv run python api_server/mcp_server/mcp_server.py
```

### Inspect and select scenarios / traffic lights (TL IDs)

```bash
# list available scenarios under ./scenarios
uv run python api_server/mcp_server/mcp_server.py --list-scenarios

# list TL IDs in a scenario
uv run python api_server/mcp_server/mcp_server.py --scenario Doerpfeldstr_all_modes --list-tl-ids

# run a scenario (auto-pick the .sumocfg inside)
uv run python api_server/mcp_server/mcp_server.py --scenario Doerpfeldstr_all_modes

# explicitly specify a .sumocfg when a directory contains multiple configs
uv run python api_server/mcp_server/mcp_server.py --sumocfg scenarios/Doerpfeldstr_all_modes/all_modes.sumocfg

# optional: choose TL ID, disable GUI, disable auto optimization
uv run python api_server/mcp_server/mcp_server.py --scenario Doerpfeldstr_all_modes --tl-id J54 --nogui --no-auto-optimize
```

### Evaluation outputs

- Metrics CSV examples: `results/intersection_metrics_*.csv`
- Comparison notebook: `traffic_control_comparison.ipynb`
- The charts above are stored at:
  - `images/avg_saturation_comparison.png`
  - `images/avg_queue_length_comparison.png`
  - `images/avg_congestion_index_comparison.png`

## UI

<img src="images/SUMO仿真界面.png" alt="SUMO" width="50%" height="50%">


## Citation

If you use this project in your research, please cite:

```bibtex
@software{deepsignal_traffic_2025,
  title   = {DeepSignal (Traffic): LLM-based Traffic Signal Control with SUMO + MCP},
  author  = {AIMS Laboratory},
  year    = {2025},
  url     = {https://github.com/AIMSLaboratory/DeepSignal}
}
```
