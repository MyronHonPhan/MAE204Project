import modern_robotics as mr
import numpy as np

# help(mr.ScrewTrajectory)
# help(mr.FKinSpace)

N = 10

Tse_initial = np.array([[0, 0, 1, 323.5756],
                    [-1, 0, 0, -335.5507],
                    [0, -1, 0, 237.0000],
                    [0, 0, 0, 1]])

Tsc_initial = np.array([[1, 0, 0, 450],
                    [0, 1, 0, -300],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

Tsc_final = np.array([[0, -1, 0, 0],
                    [1, 0, 0, 100],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

Tce_grasp = np.array([[0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 20],
                    [0, 0, 0, 1]])

Tce_standoff = np.array([[0, 0, 1, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 70],
                    [0, 0, 0, 1]])


def SE3Flattened(T):
    # decompose
    R = T[:3,:3]
    t = T[:3,3]
    
    # flatten
    R_flattened = R.flatten()
    t_flattened = t.flatten()
    
    return np.hstack((R_flattened,t_flattened))
    

def trajectoryGenerator(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, N):
    method = 5
    Tf = 5
    
    # standoff positions at initial and final cube configurations
    Tse_standoff = Tsc_initial @ Tce_standoff
    Tse_standoff_final = Tsc_final @ Tce_standoff
    
    # grasp positions at initial and final cube configurations
    Tse_grasp = Tsc_initial @ Tce_grasp
    Tse_grasp_final = Tsc_final @ Tce_grasp
    # initial config to standoff trajectory
    first_trajectory = mr.ScrewTrajectory(Tse_initial, Tse_standoff, Tf, N, method)
    # standoff to grasp trajectory
    second_trajectory = mr.ScrewTrajectory(Tse_standoff, Tse_grasp, Tf, N, method)
    # grasp to standoff trajectory
    third_trajectory = mr.ScrewTrajectory(Tse_grasp, Tse_standoff, Tf, N, method)
    # standoff to final config trajectory
    fourth_trajectory = mr.ScrewTrajectory(Tse_standoff, Tse_standoff_final, Tf, N, method)
    # standoff to standoff final config trajectory
    fifth_trajectory = mr.ScrewTrajectory(Tse_standoff_final, Tse_grasp_final, Tf, N, method)
    # final config to standoff trajectory
    final_trajectory = mr.ScrewTrajectory(Tse_grasp_final, Tse_standoff_final, Tf, N, method) 
    
    whole_trajectory = first_trajectory + second_trajectory + third_trajectory + fourth_trajectory + fifth_trajectory + final_trajectory
    
    gripper_state_1 = [0]*N
    gripper_state_2 = [0]*N
    gripper_state_3 = [1]*N
    gripper_state_4 = [1]*N
    gripper_state_5 = [1]*N
    gripper_state_6 = [0]*N
    
    gripper_trajectory = gripper_state_1 + gripper_state_2 + gripper_state_3 + gripper_state_4 + gripper_state_5 + gripper_state_6
    
    return np.asarray(whole_trajectory), np.asarray(gripper_trajectory)
    
def turnTrajectoryIntoCSV(pose_trajectory, gripper_trajectory):
    for i in range(pose_trajectory.shape[0]):
        # flatten them into rows
        T_flattened = SE3Flattened(pose_trajectory[i,...])
        gripper_state = gripper_trajectory[i]
        
        combined = np.hstack((T_flattened, np.array([gripper_state])))
        
        if i == 0:
            data = combined
        else:
            data = np.vstack((data, combined))
            
    np.savetxt("elements.csv", data, delimiter=",",fmt="%.2f")

pose_trajectory , gripper_trajectory = trajectoryGenerator(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, N)
turnTrajectoryIntoCSV(pose_trajectory, gripper_trajectory)