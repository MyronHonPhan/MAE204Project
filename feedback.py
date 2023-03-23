import numpy as np
import modern_robotics as mr


def calculateJacobianBody(Slist, thetalist, Tsb):
    '''Calculate the current Jacobian in the body frame

    Parameters:
        Slist: list of screw axes along columns of 2d array of shape (6,N)
        thetalsit: A list of joint coordinates of the robot (N,)
        Tsb: the current pose of the robot end effector
    Returns:
        The body Jacobian of shape (6,N) real numbers 
    
    Example use:
        SLIST = np.array([[0, 0, 1, -300, 0, 0],
                      [0, 1, 0, -240, 0, 0],
                      [0, 1, 0, -240, 0, 244],
                      [0, 1, 0, -240, 0, 457],
                      [0, 0, -1, 169, 457, 0],
                      [0, 1, 0, -155, 0, 457]]).T
        M = np.array([[1, 0, 0, 457], [0, 1, 0, 78], [0, 0, 1, 155], [0, 0, 0, 1]])

        thetalist = np.array([-PI/6, -PI/2, PI/2, -PI/2, -PI/2, 5 * PI/6])
        Tse = mr.FKinSpace(M, SLIST, thetalist)

        JacobianEndEffector = calculateJacobianBody(SLIST, thetalist, Tse)
    Output:
        array([[-1.51659286e-16, -8.66025404e-01, -8.66025404e-01,
        -8.66025404e-01,  5.00000000e-01,  0.00000000e+00],
       [-1.00000000e+00,  1.11022302e-16,  1.11022302e-16,
         1.11022302e-16,  0.00000000e+00,  1.00000000e+00],
       [-4.06369831e-17,  5.00000000e-01,  5.00000000e-01,
         5.00000000e-01,  8.66025404e-01,  0.00000000e+00],
       [-3.23575570e+02, -1.50000000e+00, -1.23500000e+02,
        -1.23500000e+02, -2.13908275e+02,  1.50047908e-14],
       [ 4.76285678e-14,  2.98000000e+02,  2.98000000e+02,
         8.50000000e+01,  2.84217094e-14, -6.31088724e-30],
       [ 3.55506721e+01, -2.59807621e+00, -2.13908275e+02,
        -2.13908275e+02,  1.23500000e+02,  1.88111102e-14]])

    '''
    Js = mr.JacobianSpace(Slist, thetalist)
    AdjointTransform = np.linalg.inv(Tsb)
    AdjointMatrix = mr.Adjoint(AdjointTransform)

    Jb = AdjointMatrix @ Js

    return Jb


def feedback(Tse, Tse_d, Tse_d_next, kp, ki, dt):
    '''Performs PI feedback/feed-forward control to get the commanded next twist of 
    the end effector

    Parameters:
        Tse: The current pose of the end effector in SE(3)
        Tse_d: The current desired pose of the end-effector in SE(3)
        Tse_d: The next desired pose of the end-effector in SE(3)
        kp: control gain param
        ki: control gain param
        dt: the length of the time step in seconds

    Returns:
        Twist of the end effector in body frame that will get carried out 
        
    
    Example use:   
        Tse_d = np.array(
            [[0, 0, 1, 300], [-1, 0, 0, -300],[0, -1, 0, 237], [0, 0, 0, 1]])
        Tse_d_next = np.array(
            [[0, 0, 1, 290], [-1, 0, 0, -290], [0, -1, 0, 237], [0, 0, 0, 1]])
        kp = 1
        ki = 0
        dt = 0.01
        feedback(Tse, Tse_d, Tse_d_next, kp, ki, dt)
    Output:
        array([ 4.06369831e-17 -5.55111512e-17 -1.51659286e-16 -1.03555067e+03
                1.60699744e-13 -1.02357557e+03])
    '''
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
    '''Gets the joint velocities based on the Jacobian and the desired twist

    Parameters:
        JacobianEndEffector
        TwistEndEffector

    Returns:
        List of velocities that will be applied for the next time step
        
    Example use:
        JacobianEndEffector = calculateJacobianBody(...)
        TwistEndEffector = feedback(...)
        
        retrieveJointVelocities(JacobianEndEffector, TwistEndEffector)
    
    Output:
        array([ 1.29203156e+00 -5.06131823e+00  5.06131823e+00  2.18652541e-13
            -5.48865773e-14  1.29203156e+00])
    '''
    return np.linalg.pinv(JacobianEndEffector, rcond=1e-5) @ TwistEndEffector


if __name__ == "__main__":
    # constants
    PI = np.pi
    SLIST = np.array([[0, 0, 1, -300, 0, 0],
                      [0, 1, 0, -240, 0, 0],
                      [0, 1, 0, -240, 0, 244],
                      [0, 1, 0, -240, 0, 457],
                      [0, 0, -1, 169, 457, 0],
                      [0, 1, 0, -155, 0, 457]]).T
    M = np.array([[1, 0, 0, 457], [0, 1, 0, 78], [0, 0, 1, 155], [0, 0, 0, 1]])

    # configuration for testing

    # initial config
    thetalist = np.array([-PI/6, -PI/2, PI/2, -PI/2, -PI/2, 5 * PI/6])
    Tse = mr.FKinSpace(M, SLIST, thetalist)

    # desired trajectory
    Tse_d = np.array([[0, 0, 1, 300], [-1, 0, 0, -300],
                     [0, -1, 0, 237], [0, 0, 0, 1]])
    Tse_d_next = np.array(
        [[0, 0, 1, 290], [-1, 0, 0, -290], [0, -1, 0, 237], [0, 0, 0, 1]])

    # control gains
    kp = 1
    ki = 0

    # time step
    dt = 0.01

    JacobianEndEffector = calculateJacobianBody(SLIST, thetalist, Tse)

    # start test

    TwistEndEffector = feedback(Tse, Tse_d, Tse_d_next, kp, ki, dt)
    JointVelocities = retrieveJointVelocities(
        JacobianEndEffector, TwistEndEffector)
