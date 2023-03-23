import numpy as np
from nextSate import nextState
if __name__=="__main__":
    PI = np.pi
    t_step = 0.01
    position = np.array([-PI/6,-PI/2,PI/2,-PI/2,-PI/2,5*PI/6])
    const_velo = np.array([PI/2,0,0,0,0,PI])
    max_velos = np.array([PI,PI,PI,PI,PI,PI])
    with open('next_state_joint_angs.csv','w') as csv_file:
        for _ in range(100):
            position = nextState(position,const_velo,t_step,max_velos)
            pos_str = np.array2string(position,separator=',')[1:-2]
            csv_file.write(pos_str+',1.0\n')

            
        