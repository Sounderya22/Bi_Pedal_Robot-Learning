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

plans = load_footstep_plans("mujoco_env/footstep_plans.txt")
walking_speed = 0.5
dt = 0.033

min_vx = 1.0
max_vx = -1.0

for plan_idx, footsteps in enumerate(plans):
    for i in range(len(footsteps) - 1):
        x0, y0, th0 = footsteps[i]
        x1, y1, th1 = footsteps[i + 1]
        
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        step_duration = distance / walking_speed if distance > 0 else 0.5
        steps = max(1, int(step_duration / dt))
        
        for t in range(steps):
            alpha = t / steps
            x = (1 - alpha) * x0 + alpha * x1
            y = (1 - alpha) * y0 + alpha * y1
            
            # Handle angle wrap around for interpolation
            theta_diff = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
            theta = th0 + alpha * theta_diff
            
            # Look ahead
            next_alpha = (t + 1) / steps
            next_x = (1 - next_alpha) * x0 + next_alpha * x1
            next_y = (1 - next_alpha) * y0 + next_alpha * y1
            
            vx_global = (next_x - x) / dt
            vy_global = (next_y - y) / dt
            
            local_vx = vx_global * math.cos(theta) + vy_global * math.sin(theta)
            local_vy = -vx_global * math.sin(theta) + vy_global * math.cos(theta)
            
            if local_vx < min_vx: min_vx = local_vx
            if local_vx > max_vx: max_vx = local_vx
            
            if local_vx < -0.1:
                print(f"Plan {plan_idx}, step {i}: local_vx = {local_vx:.3f}, th0={th0:.3f}, th1={th1:.3f}, dx={x1-x0:.3f}, dy={y1-y0:.3f}")

print(f"Min local_vx: {min_vx:.3f}, Max local_vx: {max_vx:.3f}")

