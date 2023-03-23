from trajectoryGeneration import trajectoryGenerator, turnTrajectoryIntoCSV
from feedback import retrieveJointVelocities, feedback, calculateJacobianBody
from nextSate import nextState
import matplotlib.pyplot as plt
import modern_robotics as mr
import numpy as np

PI = np.pi

# relative poses to the cube in all cases for standoff and grasping
TCE_STANDOFF = np.array([[0, 0, 1, 0],
                         [-1, 0, 0, 0],
                         [0, -1, 0, 70],
                         [0, 0, 0, 1]])

TCE_GRASP = np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 20],
                      [0, 0, 0, 1]])

N = 251  # number of intermediate steps in trajectory positions
DT = 0.01  # amount of time between each iteration (seconds)

# screw axes list for our given UR3e robot
SLIST = np.array([[0, 0, 1, -300, 0, 0],
                  [0, 1, 0, -240, 0, 0],
                  [0, 1, 0, -240, 0, 244],
                  [0, 1, 0, -240, 0, 457],
                  [0, 0, -1, 169, 457, 0],
                  [0, 1, 0, -155, 0, 457]]).T
# zeros position for our given UR3e robot
M = np.array([[1, 0,  0, 457],
              [0, 1,  0, 78],
              [0, 0, 1, 155],
              [0, 0,  0, 1]])


def finalProject(MODE):
    ''' Main container function for the final project putting together all three
    parts of trajectory generation, feedback control, and forward dynamics. PLots 
    error logs and dumps output joint angles to a csv file.

    Parameters:
        MODE - Mode to run the function in
            1 - Best test case
            2 - Overshoot test case
            3 - New Task test case

    Output: None
        *writes joint angles to file called 'TESTNAME_finalProject.csv'
    '''
    # original position of the end effector
    Tse_initial = np.array([[0, 1, 0, 247],
                            [1, 0, 0, -169],
                            [0, 0, -1, 782],
                            [0, 0, 0, 1]])
    # initial joint angles of e-e
    position = np.array([0.0012750106140693163, -1.5718041310748543,
                        PI/6, -1.5727349624970302, -1.572071337408965, 0.0])

    # set parameters based on the mode selected
    if MODE == 0:
        test_name = "Best Case"
        kp = 10
        ki = 1
        Tsc_final = np.array([[0, -1, 0, 0],
                              [1, 0, 0, 100],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        Tsc_initial = np.array([[1, 0, 0, 450],
                                [0, 1, 0, -300],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    elif MODE == 1:
        test_name = "Overshoot Case"
        kp = 4
        ki = 0
        Tsc_final = np.array([[0, -1, 0, 0],
                              [1, 0, 0, 100],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        Tsc_initial = np.array([[1, 0, 0, 450],
                                [0, 1, 0, -300],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    elif MODE == 2:
        test_name = "New Task Case"
        Tsc_initial = np.array([[1, 0, 0, 550],
                                [0, 1, 0, -400],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        Tsc_final = np.array([[0, -1, 0, 0],
                              [1, 0, 0, 125],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        kp = 10
        ki = 1
    else:
        return print("Not a valid mode. Valid modes are 0, 1, and 2")

    # max velocities
    max_velos = np.array([10000*PI, 10000*PI, 10000*PI,
                         10000*PI, 10000*PI, 10000*PI])

    # generate trajectory for our task
    pose_trajectory, gripper_trajectory = trajectoryGenerator(
        Tse_initial, Tsc_initial, Tsc_final, TCE_GRASP, TCE_STANDOFF, N)
    turnTrajectoryIntoCSV(pose_trajectory, gripper_trajectory)

    # initial Tse
    Tse = pose_trajectory[0, ...]
    if MODE == 2:
        Tse = mr.FKinSpace(M, SLIST, position)

    # initialize trajectory log and error logs
    Tse_trajectory = np.zeros((pose_trajectory.shape[0], 4, 4))
    ErrorTwist_trajectory = np.zeros((pose_trajectory.shape[0]-1, 6))
    Tse_trajectory[0, ...] = Tse
    ErrorTwist_trajectory[0, ...] = np.zeros((6,))
    error_trajectory = np.zeros((pose_trajectory.shape[0]-1, 3))

    # initial joint angles
    position_trajectory = np.zeros((pose_trajectory.shape[0], 6))
    position_trajectory[0, :] = position

    # main loop to
    for i in range(pose_trajectory.shape[0]-1):
        # desired trajectories
        Tse_d = pose_trajectory[i, ...]
        Tse_d_next = pose_trajectory[i+1, ...]

        # calculate error twist, end effector jacobian and then joint velocities at time i
        TwistEndEffector = feedback(Tse, Tse_d, Tse_d_next, kp, ki, DT)
        JacobianEndEffector = calculateJacobianBody(SLIST, position, Tse)
        JointVelocities = retrieveJointVelocities(
            JacobianEndEffector, TwistEndEffector)

        # calculate next state and compute forward dynamics for new configuration
        position = nextState(position, JointVelocities, DT, max_velos)
        position_trajectory[i+1, :] = position
        Tse = mr.FKinSpace(M, SLIST, position)

        # log Tse and error twist into trajectory
        Tse_trajectory[i+1, ...] = Tse
        ErrorTwist_trajectory[i, ...] = TwistEndEffector

        error = np.abs(np.linalg.norm(Tse_d_next.flatten()-Tse.flatten()))
        error_trajectory[i, :] = error

    # plot the total error over time of the trajectory
    plt.figure(1)
    plt.plot(error_trajectory)
    plt.xlabel("Iteration number")
    plt.ylabel("Frobenius Norm of Difference Error")
    plt.title(
        test_name + ": Frobenius Norm of Difference Error VS Iteration Number")
    plt.show()

    # plot the error in x, y, and z orientation over time
    plt.figure(2)
    plt.plot(ErrorTwist_trajectory[:, 0])
    plt.plot(ErrorTwist_trajectory[:, 1])
    plt.plot(ErrorTwist_trajectory[:, 2])
    plt.legend(["E_Rx", "E_Ry", "E_Rz"])
    plt.xlabel("Iteration number")
    plt.ylabel("Twist Error")
    plt.title(test_name + ": Error Angular Twist VS Iteration Number")
    plt.show()

    # plot the error in x, y, and z position over time
    plt.figure(3)
    plt.plot(ErrorTwist_trajectory[:, 3])
    plt.plot(ErrorTwist_trajectory[:, 4])
    plt.plot(ErrorTwist_trajectory[:, 5])
    plt.legend(["E_x", "E_y", "E_z"])
    plt.xlabel("Iteration number")
    plt.ylabel("Linear Error")
    plt.title(test_name + ": Error Linear Twist VS Iteration Number")
    plt.show()

    # save all joint angles and gripper values to a csv to be simulated in Gazebo
    data = np.hstack((position_trajectory, gripper_trajectory[:, None]))
    np.savetxt(test_name.replace(" ","_")+"_finalProject.csv", data, delimiter=",", fmt="%.2f")


def best_case():
    ''' Run the "best" test case as decribed in the write up'''
    finalProject(0)


def overshoot_case():
    ''' Run the "overshoot" test case as decribed in the write up'''
    finalProject(1)


def new_task_case():
    ''' Run the "new_task" test case, making a new task for the robot arm'''
    finalProject(2)


if __name__ == "__main__":
    best_case()
    overshoot_case()
    new_task_case()
