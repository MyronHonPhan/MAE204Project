import modern_robotics as mr
import numpy as np


def SE3Flattened(T):
    '''Given a transformation matrix in SE(3) returns a flattened version of it to 
    be used for the Matlab trajectory animation script

    Parameters:
        T: The given transformation matrix

    Outputs:
        A flattened array of the Rotation and Position values

    Example Input:
    T = = np.array([[0, 1, 0, 247],
                    [1, 0, 0, -169],
                    [0, 0, -1, 782],
                    [0, 0, 0, 1]])
    Output:
    array([0.00,1.00,0.00,1.00,0.00,0.00,0.00,0.00,-1.00,247.00,-169.00,782.00,0.00]) 
    '''
    # decompose
    R = T[:3, :3]
    t = T[:3, 3]

    # flatten
    R_flattened = R.flatten()
    t_flattened = t.flatten()

    return np.hstack((R_flattened, t_flattened))


def trajectoryGenerator(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, N):
    ''' Computes a trajectory for our given task using the key poses of the task
    space given as Transformation matrices in SE(3)
    Task: Pick up a single cube and move it to another location

    Parameters:
        Tse_initial: The original pose of the end effector
        Tsc_initial: The original pose of the cube  
        Tsc_final: The desired final pose of the cube
        Tce_grasp: The pose of the robot in relation to the cube as its grasping it
        Tce_standoff: The pose of the robot at standoff in relation to the cube
        N: The number of points in the trajectory between each position given 

    Outputs: (whole_trajectory, gripper_trajectory)
        whole_trajectory: The list of 6*N poses in SE(3) for the robot end effector
                        along the desired trajectory 
                        shape: (6*N,4,4)
        gripper trajectory: The list of 6*N binary values depending on whether the 
                        robot gripper should be open(0) or closed(1)
                        shape: (6*N,)

    Example Input:
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
    N = 10

    Output:
    whole trajectory:
        array([[   0.    ,    0.    ,    1.    ,  323.5756],
        [  -1.    ,    0.    ,    0.    , -335.5507],
        [   0.    ,   -1.    ,    0.    ,  237.    ],
        [   0.    ,    0.    ,    0.    ,    1.    ]])
        ...
        ...
        array([[  1.,   0.,   0.,   0.],
        [  0.,   0.,   1., 100.],
        [  0.,  -1.,   0.,  70.],
        [  0.,   0.,   0.,   1.]])   
    gripper trajectory:
        array([0,0,0,0,0,0,0,0,0,0,....1,1,1,1,1,1,1,1,1,1,...,0,0,0,0,0,0,0,0,0,0])    
    '''

    method = 5
    Tf = 5

    # standoff positions at initial and final cube configurations
    Tse_standoff = Tsc_initial @ Tce_standoff
    Tse_standoff_final = Tsc_final @ Tce_standoff

    # grasp positions at initial and final cube configurations
    Tse_grasp = Tsc_initial @ Tce_grasp
    Tse_grasp_final = Tsc_final @ Tce_grasp
    # initial config to standoff trajectory
    first_trajectory = mr.ScrewTrajectory(
        Tse_initial, Tse_standoff, Tf, N, method)
    # standoff to grasp trajectory
    second_trajectory = mr.ScrewTrajectory(
        Tse_standoff, Tse_grasp, Tf, N, method)
    # grasp to standoff trajectory
    third_trajectory = mr.ScrewTrajectory(
        Tse_grasp, Tse_standoff, Tf, N, method)
    # standoff to final config trajectory
    fourth_trajectory = mr.ScrewTrajectory(
        Tse_standoff, Tse_standoff_final, Tf, N, method)
    # standoff to standoff final config trajectory
    fifth_trajectory = mr.ScrewTrajectory(
        Tse_standoff_final, Tse_grasp_final, Tf, N, method)
    # final config to standoff trajectory
    final_trajectory = mr.ScrewTrajectory(
        Tse_grasp_final, Tse_standoff_final, Tf, N, method)

    whole_trajectory = first_trajectory + second_trajectory + \
        third_trajectory + fourth_trajectory + fifth_trajectory + final_trajectory
    # gripper is the same during the entire trajectory segment
    gripper_state_1 = [0]*N
    gripper_state_2 = [0]*N
    gripper_state_3 = [1]*N
    gripper_state_4 = [1]*N
    gripper_state_5 = [1]*N
    gripper_state_6 = [0]*N

    gripper_trajectory = gripper_state_1 + gripper_state_2 + \
        gripper_state_3 + gripper_state_4 + gripper_state_5 + gripper_state_6

    return np.asarray(whole_trajectory), np.asarray(gripper_trajectory)


def turnTrajectoryIntoCSV(pose_trajectory, gripper_trajectory, filename="elements.csv"):
    ''' Turns a trajectory of poses and gripper values into a CSV file for the Matlab
    animation script

    Parameters:
        pose_trajectory: 3d numpy array of poses to be converted
        gripper_trajectory: 1D numpy array of gripper values of the same size as the
                            first dimension of pose_trajectory
        filename: name of csv file to be written to (DEFAULT:"elements.csv")

    Outputs: None
        *writes to a file given by filename parameter
    '''
    for i in range(pose_trajectory.shape[0]):
        # flatten them into rows
        T_flattened = SE3Flattened(pose_trajectory[i, ...])
        gripper_state = gripper_trajectory[i]
        # add gripper position to end of csv line
        combined = np.hstack((T_flattened, np.array([gripper_state])))

        if i == 0:
            data = combined
        else:
            data = np.vstack((data, combined))

    np.savetxt(filename, data, delimiter=",", fmt="%.2f")


if __name__ == "__main__":
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

    pose_trajectory, gripper_trajectory = trajectoryGenerator(
        Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, N)
    turnTrajectoryIntoCSV(pose_trajectory, gripper_trajectory)
