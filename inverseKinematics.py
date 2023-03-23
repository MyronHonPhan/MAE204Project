import modern_robotics as mr
import numpy as np

# help(mr.IKinSpace)

PI = np.pi
# Tse_initial = np.array([[0, 0, 1, 323.5756],
#                         [-1, 0, 0, -335.5507],
#                         [0, -1, 0, 237.0000],
#                         [0, 0, 0, 1]])

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

def inverseKinematics(Tse_initial, thetalist0, eomg, ev, Slist, M):
        theta_out, converged = mr.IKinSpace(Slist, M, Tse_initial, thetalist0, eomg, ev)
        print("converged:",converged)
        thetas_initial = []
        for joint_ang in theta_out:
            while joint_ang > PI:
                joint_ang -= 2*PI
            while joint_ang < -PI:
                joint_ang += 2*PI
            thetas_initial.append(joint_ang)
        return thetas_initial

print(inverseKinematics(Tse_initial, thetalist0, eomg, ev))