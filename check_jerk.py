import math
import numpy as np

# We'll just look at the velocities with and without my fix to show the user
def test_traj_oscillation():
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
    dt = 0.033
    walking_speed = 0.5
    
    print("=== Original Zig-Zag Trajectory ===")
    for i in range(4):
        x0, y0, th0 = footsteps[i]
        x1, y1, th1 = footsteps[i + 1]
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        step_duration = distance / walking_speed if distance > 0 else 0.5
        steps = max(1, int(step_duration / dt))
        t = 0
        alpha = t / steps
        x = (1 - alpha) * x0 + alpha * x1
        y = (1 - alpha) * y0 + alpha * y1
        theta_diff = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
        theta = th0 + alpha * theta_diff
        
        next_alpha = (t + 1) / steps
        next_x = (1 - next_alpha) * x0 + next_alpha * x1
        next_y = (1 - next_alpha) * y0 + next_alpha * y1
        vx_global = (next_x - x) / dt
        vy_global = (next_y - y) / dt
        local_vx = vx_global * math.cos(theta) + vy_global * math.sin(theta)
        local_vy = -vx_global * math.sin(theta) + vy_global * math.cos(theta)
        print(f"Step {i}: local_vy flips to {local_vy:+.3f} m/s")

    print("\n=== Proposed Smooth Trajectory (Midpoints) ===")
    base_waypoints = [(0.0, 0.0, footsteps[0][2])]
    for i in range(len(footsteps) - 1):
        x0, y0, th0 = footsteps[i]
        x1, y1, th1 = footsteps[i + 1]
        x = (x0 + x1) / 2.0
        y = (y0 + y1) / 2.0
        theta_diff = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
        theta = th0 + theta_diff / 2.0
        base_waypoints.append((x, y, theta))
        
    for i in range(4):
        x0, y0, th0 = base_waypoints[i]
        x1, y1, th1 = base_waypoints[i + 1]
        distance = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        step_duration = distance / walking_speed if distance > 0 else 0.5
        steps = max(1, int(step_duration / dt))
        t = 0
        alpha = t / steps
        x = (1 - alpha) * x0 + alpha * x1
        y = (1 - alpha) * y0 + alpha * y1
        theta_diff = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
        theta = th0 + alpha * theta_diff
        
        next_alpha = (t + 1) / steps
        next_x = (1 - next_alpha) * x0 + next_alpha * x1
        next_y = (1 - next_alpha) * y0 + next_alpha * y1
        vx_global = (next_x - x) / dt
        vy_global = (next_y - y) / dt
        local_vx = vx_global * math.cos(theta) + vy_global * math.sin(theta)
        local_vy = -vx_global * math.sin(theta) + vy_global * math.cos(theta)
        print(f"Step {i}: local_vy is smoothly {local_vy:+.3f} m/s")

test_traj_oscillation()
