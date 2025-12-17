# DeepSignal (SUMO + MCP) — Traffic Signal Control via LLM

[中文 README](README_zh.md)

DeepSignal is our in-house fine-tuned large language model for **traffic-signal control**. The current release is **DeepSignal-4B**.

- **Model (Hugging Face)**: [`AIMS2025/DeepSignal`](https://huggingface.co/AIMS2025/DeepSignal)


This repository also contains a SUMO-based simulation stack and an MCP server to run closed-loop interaction between the LLM and traffic simulations.

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
| Chengdu | `sumo_llm` | `osm.sumocfg` | Eval (test-only) | Test-only; NOT used in fine-tuning/training |

## Evaluation metrics

We evaluate DeepSignal in **SUMO simulation** using intersection-level metrics computed from the simulator:

- **Avg Saturation** (`average_saturation`)
- **Avg Queue Length** (`average_queue_length`)
- **Max Saturation** (`max_saturation`)
- **Max Queue Length** (`max_queue_length`)
- **Congestion Index (0–1)** (`congestion_index`)
- **Congestion Level** (`congestion_level`) and its distribution (%)


## Results (metrics comparison)

### Performance Metrics Comparison by Model

| Model | Avg Saturation | Avg Queue Length | Max Saturation | Max Queue Length | Avg Congestion Index |
|---|---:|---:|---:|---:|---:|
| [`Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | 0.1663 | 5.8604 | 0.1663 | 5.8604 | 0.1625 |
| DeepSignal-4B (Ours) | 0.1657 | 5.8301 | 0.1657 | 5.8301 | 0.1752 |
| [`LightGPT-8B-Llama3`](https://huggingface.co/lightgpt/LightGPT-8B-Llama3) | 0.1538 | 5.7688 | 0.1538 | 5.7688 | 0.2086 |
| Qwen3-4B-SFT | 0.1604 | 6.0021 | 0.1604 | 6.0021 | 0.2093 |
| [`Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 0.2152 | 8.2083 | 0.2152 | 8.2083 | 0.2522 |
| Max Pressure | 0.2059 | 8.1034 | 0.2059 | 8.1034 | 0.2556 |
| [`GPT-OSS-20B`](https://huggingface.co/openai/gpt-oss-20b) | 0.2591 | 10.4292 | 0.2591 | 10.4292 | 0.3175 |


### Congestion Level Distribution by Model (%)

| Model | Light congestion | Smooth | Very smooth |
|---|---:|---:|---:|
| DeepSignal-4B (Ours) | 0.00 | 12.00 | 88.00 |
| [`GPT-OSS-20B`](https://huggingface.co/openai/gpt-oss-20b) | 2.00 | 53.33 | 44.67 |
| [`LightGPT-8B-Llama3`](https://huggingface.co/lightgpt/LightGPT-8B-Llama3) | 0.00 | 21.00 | 79.00 |
| Max Pressure | 0.00 | 36.44 | 63.56 |
| [`Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | 0.00 | 10.00 | 90.00 |
| [`Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 2.33 | 32.00 | 65.67 |
| Qwen3-4B-SFT | 0.00 | 23.33 | 76.67 |

### Visualization

![Metrics Comparison](images/metrics_comparison_bars.png)

## Model files (GGUF) and local inference

If you are looking for GGUF files for local inference (`llama.cpp` / LM Studio), check the model card in Hugging Face and the packaging notes under `hf/`.

Example (llama.cpp):

```bash
llama-cli -m DeepSignal-4B_V1.F16.gguf -p "Summarize the traffic state and suggest a signal timing adjustment."
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
- The bar chart above is stored at: `images/metrics_comparison_bars.png`

## UI

<img src="images/SUMO仿真界面.png" alt="SUMO" width="50%" height="50%">
