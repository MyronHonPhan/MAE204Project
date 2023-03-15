import numpy as np
import modern_robotics as mr


def calculateJacobianBody(Slist, thetalist, Tsb):
    Js = mr.JacobianSpace(Slist, thetalist)
    AdjointTransform = np.linalg.inv(Tsb)
    AdjointMatrix = mr.Adjoint(AdjointTransform)
    
    Jb = AdjointMatrix @ Js
    
    return Jb

def feedback(Tse, Tse_d, Tse_d_next, kp, ki, dt):
    # SE(3), calculate error transform
    errorTransform = np.linalg.inv(Tse) @ Tse_d     
    # se(3), log map
    errorTwistMatrix = mr.MatrixLog6(errorTransform)
    # R^6
    errorTwistVector = mr.se3ToVec(errorTwistMatrix)
    
    # SE(3) feedforward term, captures how we want to move
    desiredTransform = np.linalg.inv(Tse_d) @ Tse_d_next
    # se(3)
    desiredTwistMatrix = mr.MatrixLog6(desiredTransform)
    # R^6, divide by dt as a discrete time analog
    desiredTwistVector = mr.se3ToVec(desiredTwistMatrix) / dt
    
    # initialize control gain matrices
    Kp = np.eye(6) * kp
    Ki = np.eye(6) * ki
    
    # calculate adjoint to bring the desired twist into the e-e frame
    adjointTransform = errorTransform
    adjointMatrix = mr.Adjoint(adjointTransform)
    
    # bring it all together, calculate end effector twist
    endEffectorTwist = adjointMatrix @ desiredTwistVector + Kp @ errorTwistVector \
        + Ki @ errorTwistVector * dt
        
    return endEffectorTwist

def retrieveJointVelocities(JacobianEndEffector, TwistEndEffector):
    return np.linalg.pinv(JacobianEndEffector) @ TwistEndEffector

################ TESTING ###################

# constants
pi = np.pi

Slist = np.array([[0 , 0 , 1 , -300 , 0 , 0],
[0 , 1 , 0 , -240 , 0 , 0],
[0 , 1 , 0 , -240 , 0 , 244],
[0 , 1 , 0 , -240 , 0 , 457],
[0 , 0 , -1 , 169 , 457 , 0],
[0 , 1 , 0 , -155 , 0 , 457]]).T

M = np.array([[1, 0, 0, 457] , [0, 1, 0, 78] , [0, 0, 1, 155] , [0, 0, 0, 1]])

# configuration for testing

# initial config
thetalist = np.array([-pi/6, -pi/2, pi/2, -pi/2, -pi/2, 5 * pi/6])
Tse = mr.FKinSpace(M, Slist, thetalist)

# desired trajectory
Tse_d = np.array([[0, 0, 1, 300] , [-1, 0, 0, -300] , [0, -1, 0, 237] , [0, 0, 0, 1]])
Tse_d_next = np.array([[0, 0, 1, 290] , [-1, 0, 0, -290] , [0, -1, 0, 237] , [0, 0, 0, 1]])

# control gains
kp = 1
ki = 0

# time step
dt = 0.01

JacobianEndEffector = calculateJacobianBody(Slist, thetalist, Tse)

# start test

TwistEndEffector = feedback(Tse, Tse_d, Tse_d_next, kp, ki, dt)
JointVelocities = retrieveJointVelocities(JacobianEndEffector, TwistEndEffector)