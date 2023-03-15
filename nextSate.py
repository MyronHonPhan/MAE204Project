import modern_robotics as mr
import numpy as np

def nextState(curr_state, joint_velos, timestep, max_joint_velos):
# curr_state: current state of the robot (the 6 joint angles )θ1... θ6
# joint_velos: joint velocities (6 variables, )θ ̇ 𝑖
# timestep: timestep size (1 parameter)∆𝑡 = 0. 01
# max_joint_velos: maximum joint velocity magnitudes (1 parameter)
    velo_magnitudes = np.min(np.vstack((np.abs(joint_velos),max_joint_velos)),axis=0)
    velos = velo_magnitudes * np.sign(joint_velos)
    return np.array(curr_state) + velos*timestep