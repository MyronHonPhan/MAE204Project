from trajectoryGeneration import *
from feedback import *
from nextSate import *
import matplotlib.pyplot as plt

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
error_trajactory = np.zeros((pose_trajectory.shape[0]-1,3))

# initial joint angles
position = np.array([-pi/6,-pi/2,pi/2,-pi/2,-pi/2,5*pi/6])

# control grains
kp = 60
ki = 50

for i in range(1499):
    # desired trajectories
    Tse_d = pose_trajectory[i,...]
    Tse_d_next = pose_trajectory[i+1,...]

    # calculate error twist, end effector jacobian and then joint velocities at time i
    TwistEndEffector = feedback(Tse, Tse_d, Tse_d_next, kp, ki, dt)
    JacobianEndEffector = calculateJacobianBody(Slist, position, Tse)
    JointVelocities = retrieveJointVelocities(JacobianEndEffector, TwistEndEffector)
    
    # calculate next state and compute forward dynamics for new configuration
    position = nextState(position, JointVelocities, dt, max_velos)
    Tse = mr.FKinSpace(M, Slist, position)
    
    # log Tse and error twist into trajectory
    Tse_trajectory[i+1,...] = Tse
    ErrorTwist_trajectory[i,...] = TwistEndEffector
    
    error = np.abs(np.linalg.norm(Tse_d_next[:3,3]-Tse[:3,3]))
    error_trajactory[i,:] = error

plt.figure(1)
plt.plot(error_trajactory)
plt.show()

print("hi")