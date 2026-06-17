import math

def interpolate_trajectory(footsteps, dt=0.03, walking_speed=0.5):
    if len(footsteps) < 2:
        return footsteps

    base_waypoints = []
    # Add an initial base waypoint to start from 0,0
    x0, y0, th0 = footsteps[0]
    x1, y1, th1 = footsteps[1]
    # initial base is roughly shifted from foot 0 by (x1-x0)/2, (y1-y0)/2 ? No, just (0,0)
    base_waypoints.append((0.0, 0.0, th0))
    
    for i in range(len(footsteps) - 1):
        x0, y0, th0 = footsteps[i]
        x1, y1, th1 = footsteps[i + 1]
        x = (x0 + x1) / 2.0
        y = (y0 + y1) / 2.0
        theta_diff = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
        theta = th0 + theta_diff / 2.0
        base_waypoints.append((x, y, theta))
        
    ref_traj = []
    for i in range(len(base_waypoints) - 1):
        x0, y0, th0 = base_waypoints[i]
        x1, y1, th1 = base_waypoints[i + 1]
        
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        step_duration = distance / walking_speed if distance > 0 else 0.5
        steps = max(1, int(step_duration / dt))
        
        for t in range(steps):
            alpha = t / steps
            x = (1 - alpha) * x0 + alpha * x1
            y = (1 - alpha) * y0 + alpha * y1
            
            theta_diff = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
            theta = th0 + alpha * theta_diff
            ref_traj.append((x, y, theta))
            
    ref_traj.append(base_waypoints[-1])
    return ref_traj

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
footsteps = plans[0]
traj = interpolate_trajectory(footsteps, dt=0.033)

for i in range(len(traj) - 1):
    x, y, theta = traj[i]
    next_x, next_y, next_theta = traj[i+1]
    vx_global = (next_x - x) / 0.033
    vy_global = (next_y - y) / 0.033
    local_vx = vx_global * math.cos(theta) + vy_global * math.sin(theta)
    local_vy = -vx_global * math.sin(theta) + vy_global * math.cos(theta)
    if i % 10 == 0:
        print(f"step {i}: local_vx = {local_vx:.3f}, local_vy = {local_vy:.3f}")

