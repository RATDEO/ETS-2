import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_experiment import run_experiment

overrides = {
    'target': {'mode': 'returns'},
    'output': {'write_project_paper': False},
    'llm': {
        'api_key': 'deo',
        'base_url': 'http://192.168.1.140:9877/v1',
        'model': 'qwen3-vl-4b-gpu',
    },
}

run_dir = run_experiment(
    config_path='uk_ets/config/uk_ets_default.yaml',
    overrides=overrides,
    data_dir='uk_ets/Data_auto_uk',
)
print(f'RUN_DIR={run_dir}')
