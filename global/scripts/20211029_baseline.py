from pathlib import Path

available_gpus = (0, 1, 2, 3)
processes_per_gpu = 1
root_dir = '.'
script_name = '20211029-global-baseline'


def generate_command(exp, gpu):
    return (
            f"CUDA_VISIBLE_DEVICES={gpu} python {exp['command']} " + ' '.join(f'--{k} {v}' for k, v in exp['args'].items())
    )

settings = [
    {
        'command': 'global.py',
        'args': {
            "method": method,
            "beta": beta,
            "disentangle_fs3":'F',
            "num_test": 50,
        }
    }
    for method in ["Baseline"]
    for beta in [0.08, 0.1, 0.12]
]

print(f'num settings: {len(settings)+1}')

total_multirun_size = len(available_gpus) * processes_per_gpu
experiments_per_gpu = [[] for _ in range(total_multirun_size)]
for i, experiment in enumerate(settings):
    experiments_per_gpu[i % total_multirun_size].append(experiment)

Path('./scripts').mkdir(parents=True, exist_ok=True)
with open(f'./scripts/{script_name}.sh', 'w') as f:
    
    for i, experiment_list in enumerate(experiments_per_gpu):
        if len(experiment_list) > 0:
            
            gpu = available_gpus[i % len(available_gpus)]
            one_liner = ' '.join(
                f'{generate_command(e, gpu)} ;'
                for e in experiment_list
            )
            print(f'({one_liner}) & ', file=f)
