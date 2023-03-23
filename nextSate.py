import modern_robotics as mr
import numpy as np

def nextState(curr_state, joint_velos, timestep, max_joint_velos):
    ''' Find the next state of joint angles given the current state, the joint
    velocities, and how long the timestep is
    
    Parameters:
        curr_state: current state of the robot (the N joint angles ) θ1... θN
        joint_velos: joint velocities with shape: (N, ) (radians/sec)
        timestep: timestep size e.g. ∆𝑡 = 0.01
        max_joint_velos: maximum joint velocities with shape: (N, ) (radians/sec)
    
    Returns:
        updated joint angles of the robot with shape: (N,)
        
    Example Input:
        t_step = 0.01
        position = np.array([-PI/6,-PI/2,PI/2,-PI/2,-PI/2,5*PI/6])
        const_velo = np.array([PI/2,0,0,0,0,PI/2])
        max_velos = np.array([PI,PI,PI,PI,PI,PI])
    
    Output:
        array([-0.50789,-1.55508, 1.58650,-1.55508,-1.55508, 2.6337])
    '''
    velo_magnitudes = np.min(np.vstack((np.abs(joint_velos),max_joint_velos)),axis=0)
    velos = velo_magnitudes * np.sign(joint_velos)
    return np.array(curr_state) + velos*timestep

if __name__=="__main__":
    PI = np.pi
    t_step = 0.01
    position = np.array([-PI/6,-PI/2,PI/2,-PI/2,-PI/2,5*PI/6])
    const_velo = np.array([PI/2,0,0,0,0,PI/2])
    max_velos = np.array([PI,PI,PI,PI,PI,PI])
    # basic change in state over 1 sec and write joint-angles/gripper-state to csv 
    with open('next_state_joint_angs.csv','w') as csv_file:
        for _ in range(100):
            position = nextState(position,const_velo,t_step,max_velos)
            pos_str = np.array2string(position,separator=',')[1:-2]
            csv_file.write(pos_str+',1.0\n')