from trajectoryGeneration import trajectoryGenerator
from feedback import retrieveJointVelocities, feedback, calculateJacobianBody
from nextSate import nextState
import matplotlib.pyplot as plt
import modern_robotics as mr
import numpy as np

pi = np.pi

Slist = np.array([[0 , 0 , 1 , -300 , 0 , 0],
                    [0 , 1 , 0 , -240 , 0 , 0],
                    [0 , 1 , 0 , -240 , 0 , 244],
                    [0 , 1 , 0 , -240 , 0 , 457],
                    [0 , 0 , -1 , 169 , 457 , 0],
                    [0 , 1 , 0 , -155 , 0 , 457]]).T

M = np.array([[1, 0,  0, 457],
                [ 0, 1,  0, 78],
                [ 0, 0, 1, 155],
                [ 0, 0,  0, 1]])

Tse_initial = np.array([[0, 1, 0, 247],
                    [1, 0, 0, -169],
                    [0, 0, -1, 782],
                    [0, 0, 0, 1]])

# Tse_initial = np.array([[0, 0, 1, 323.5756],
#                     [-1, 0, 0, -335.5507],
#                     [0, -1, 0, 237.0000],
#                     [0, 0, 0, 1]])

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

N = 251

# max velocities
max_velos = np.array([10000*pi,10000*pi,10000*pi,10000*pi,10000*pi,10000*pi])

# generate trajectory
pose_trajectory , gripper_trajectory = trajectoryGenerator(Tse_initial, Tsc_initial, Tsc_final, Tce_grasp, Tce_standoff, N)

# initial Tse
Tse = pose_trajectory[0,...]

# trajectory log
Tse_trajectory = np.zeros((pose_trajectory.shape[0],4,4))
ErrorTwist_trajectory = np.zeros((pose_trajectory.shape[0]-1,6))
Tse_trajectory[0,...] = Tse
ErrorTwist_trajectory[0,...] = np.zeros((6,))
error_trajectory = np.zeros((pose_trajectory.shape[0]-1,3))

dt = 0.01

# initial joint angles

# generated from inverseKinematics.py
position = np.array([0.0012750106140693163, -1.5718041310748543, 0.0029464399820895437, -1.5727349624970302, -1.572071337408965, 0.0])
# position = np.array([-pi/6, -pi/2, pi/2, -pi/2, -pi/2, 5 * pi/6])
position_trajectory = np.zeros((pose_trajectory.shape[0],6))
position_trajectory[0,:]=position


MODE = 0
if MODE == 0: 
    kp = 10
    ki = 1
elif MODE == 1:
    kp = 1000
    ki = 50
elif MODE == 2:
    kp = 60
    ki = 50
else:
    kp = 60
    ki = 50

def best_case_1():
    pass

def overshoot_case_2():
    pass

def new_task_case_3():
    pass 

def finalProject(Tse, position, position_trajectory, error_trajectory, ErrorTwist_trajectory):
    for i in range(pose_trajectory.shape[0]-1):
        # desired trajectories
        Tse_d = pose_trajectory[i,...]
        Tse_d_next = pose_trajectory[i+1,...]

        # calculate error twist, end effector jacobian and then joint velocities at time i
        TwistEndEffector = feedback(Tse, Tse_d, Tse_d_next, kp, ki, dt)
        JacobianEndEffector = calculateJacobianBody(Slist, position, Tse)
        JointVelocities = retrieveJointVelocities(JacobianEndEffector, TwistEndEffector)
        
        # calculate next state and compute forward dynamics for new configuration
        position = nextState(position, JointVelocities, dt, max_velos)
        position_trajectory[i+1,:] = position
        Tse = mr.FKinSpace(M, Slist, position)
        
        # log Tse and error twist into trajectory
        Tse_trajectory[i+1,...] = Tse
        ErrorTwist_trajectory[i,...] = TwistEndEffector
        
        error = np.abs(np.linalg.norm(Tse_d_next[:3,3]-Tse[:3,3]))
        error_trajectory[i,:] = error
        if i % 50 == 0:
            pass

    plt.figure(1)
    plt.plot(error_trajectory)
    plt.show()
    
    data = np.hstack((position_trajectory, gripper_trajectory[:,None]))
    
    np.savetxt("finalProject.csv", data, delimiter=",",fmt="%.2f")
    
print("u the goat")
finalProject(Tse, position, position_trajectory, error_trajectory, ErrorTwist_trajectory)