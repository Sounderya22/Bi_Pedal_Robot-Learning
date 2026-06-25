import re

with open("mujoco_env/reference_generator.py", "r") as f:
    content = f.read()

# Find the update_ref_env method and add the logic to pause footstep_index
old_code = """
            self.gait_library.update_gaitlib_env(
                gait_param=self.last_ref_gaitparams, time_in_sec=time_in_sec
            )

            self.footstep_index += 1
"""

new_code = """
            self.gait_library.update_gaitlib_env(
                gait_param=self.last_ref_gaitparams, time_in_sec=time_in_sec
            )

            # Prevent the target from outrunning the robot by pausing the trajectory
            pos_err_x = (x - base_xy_g[0]) * math.cos(curr_yaw) + (y - base_xy_g[1]) * math.sin(curr_yaw)
            if pos_err_x < 0.15:
                self.footstep_index += 1
"""

# Wait, we need curr_yaw. 
# base_yaw is passed to update_ref_env!
new_code_fixed = """
            self.gait_library.update_gaitlib_env(
                gait_param=self.last_ref_gaitparams, time_in_sec=time_in_sec
            )

            # Prevent the target from outrunning the robot by pausing the trajectory
            pos_err_x = (x - base_xy_g[0]) * math.cos(base_yaw) + (y - base_xy_g[1]) * math.sin(base_yaw)
            if pos_err_x < 0.10:  # 10 cm max lead
                self.footstep_index += 1
"""

content = content.replace(old_code, new_code_fixed)

with open("mujoco_env/reference_generator.py", "w") as f:
    f.write(content)

