import math

def load_footstep_plans(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    plans = []
    for block in content.strip().split('---'):
        steps = []
        for line in block.strip().splitlines():
            if line:
                x, y, theta = map(float, line.strip().split(','))
                steps.append((x, y, theta))
        if steps:
            plans.append(steps)
    return plans

plans = load_footstep_plans('mujoco_env/footstep_plans.txt')
for i, p in enumerate(plans[:5]):
    print(f"Plan {i} starts at: {p[0]}")
