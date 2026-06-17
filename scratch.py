import math

def interpolate_trajectory(footsteps, dt=0.03, walking_speed=0.5):
    ref_traj = []
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
            
            theta_diff = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
            theta = th0 + alpha * theta_diff
            ref_traj.append((x, y, theta))
            
    ref_traj.append(footsteps[-1])
    return ref_traj

footsteps = [
    (0.0, -0.07, 0.0),
    (0.135, 0.055, -0.1963495408493614),
    (0.305, -0.225, -0.1963495408493614)
]
dt = 0.033
traj = interpolate_trajectory(footsteps, dt=dt)

for i in range(10):
    x, y, theta = traj[i]
    next_x, next_y, next_theta = traj[i+1]
    
    vx_global = (next_x - x) / dt
    vy_global = (next_y - y) / dt
    
    local_vx = vx_global * math.cos(theta) + vy_global * math.sin(theta)
    local_vy = -vx_global * math.sin(theta) + vy_global * math.cos(theta)
    
    print(f"i={i}: vx_global={vx_global:.3f}, vy_global={vy_global:.3f}, local_vx={local_vx:.3f}, local_vy={local_vy:.3f}")

