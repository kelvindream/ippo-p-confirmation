import json
import math
from pathlib import Path

ROOT = Path('/home/zrt/ai/GC-MAPPO/results/dpg_screen_v1')
METHODS = ['DPG-MAPPO', 'GC-MAPPO', 'IPPO']
SCENARIOS = ['h1', 'h2']
SEEDS = [42, 43, 44]


def load(scenario, method, seed):
    safe = method.replace(' ', '_')
    with open(ROOT / f'{scenario}_{safe}_seed{seed}' / 'evaluation_metrics.json') as fh:
        return json.load(fh)


def mean(entry, key):
    return sum(entry[key]) / len(entry[key])


rows = {}
for scenario in SCENARIOS:
    for method in METHODS:
        for seed in SEEDS:
            e = load(scenario, method, seed)
            rows[(scenario, method, seed)] = {
                'U': mean(e, 'demand_service_times'),
                'completion': mean(e, 'planned_completion_rates'),
                'teleport': sum(e['teleport_counts']),
                'conservation': all(e['demand_conservation_ok']),
            }

print('=== 相对改善 1-exp(-mean log(U_ref/U_new))，正值=DPG 更优 ===')
for ref in ['GC-MAPPO', 'IPPO']:
    for scenario in SCENARIOS:
        logs, wins = [], 0
        for seed in SEEDS:
            logs.append(math.log(rows[(scenario, ref, seed)]['U'] / rows[(scenario, 'DPG-MAPPO', seed)]['U']))
            wins += rows[(scenario, 'DPG-MAPPO', seed)]['U'] < rows[(scenario, ref, seed)]['U']
        print(f'{scenario} DPG vs {ref}: {(1 - math.exp(-sum(logs)/3))*100:+.3f}%  seed_wins={wins}/3')
    logs, wins = [], 0
    for seed in SEEDS:
        fam = sum(math.log(rows[(sc, ref, seed)]['U'] / rows[(sc, 'DPG-MAPPO', seed)]['U']) for sc in SCENARIOS) / len(SCENARIOS)
        logs.append(fam)
        wins += fam > 0
    print(f'family(H1+H2) DPG vs {ref}: {(1 - math.exp(-sum(logs)/3))*100:+.3f}%  seed_wins={wins}/3')

print()
print('=== 机制审计：residual/gate 参数范数 ===')
import torch
for scenario in SCENARIOS:
    for seed in SEEDS:
        ckpt = torch.load(ROOT / f'{scenario}_DPG-MAPPO_seed{seed}' / 'checkpoint.pt', map_location='cpu', weights_only=False)
        a = ckpt['dpg_actor']
        res = float(a['residual.2.weight'].norm()) + float(a['residual.2.bias'].norm())
        gate_keys = [k for k in a if 'gate' in k and k.endswith('weight')]
        gate_norm = sum(float(a[k].norm()) for k in gate_keys)
        print(f'{scenario} seed{seed}: residual_norm={res:.4f}  gate_norm={gate_norm:.4f}')
