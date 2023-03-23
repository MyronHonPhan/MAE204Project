import modern_robotics as mr
import numpy as np

# help(mr.IKinSpace)

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

Tse_initial = np.array([[0, 1, 0, 247+10],
                    [1, 0, 0, -169],
                    [0, 0, -1, 782],
                    [0, 0, 0, 1]])

thetalist0 = np.array([0.0012750106140693163, -1.5718041310748543, 0.0029464399820895437, -1.5727349624970302, -1.572071337408965, 0.0])
eomg = 0.01
ev = 0.001
def inverseKinematics(Tse_initial, thetalist0, eomg, ev):
    theta_out = mr.IKinSpace(Slist, M, Tse_initial, thetalist0, eomg, ev)[0]%(2*pi)
    print(mr.IKinSpace(Slist, M, Tse_initial, thetalist0, eomg, ev)[1])
    thetas_initial = []
    for joint_ang in theta_out:
        while joint_ang > pi:
            joint_ang -= 2*pi
        while joint_ang < -pi:
            joint_ang += 2*pi
        thetas_initial.append(joint_ang)
    return thetas_initial

print(inverseKinematics(Tse_initial, thetalist0, eomg, ev))