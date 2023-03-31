### Import the necessary libraries
from tqdm import tqdm
from utils import *
import time
from datetime import timedelta
from minizinc import Instance, Model, Solver
import os
import numpy as np
from glob import glob
import math

solver = Solver.lookup("gecode")

def read_ins(ins):

    with open(ins, 'r') as in_file:

        l = in_file.read().splitlines()

        w = l[0]
        n = l[1]

        cx = []
        cy = []

        for i in range(int(n)):
            line = l[i + 2].split(' ')
            cx.append(int(line[0]))
            cy.append(int(line[1]))

        return int(w), int(n), cx, cy

def write_sol(w, n, cx, cy, x_sol, y_sol, h, sol, elapsed_time):

    with open(sol, 'w+') as out:

        out.write('{} {}\n'.format(w, h))
        out.write('{}\n'.format(n))

        for i in range(n):
            out.write('{} {} {} {}\n'.format(cx[i], cy[i], x_sol[i], y_sol[i]))
        
        out.write("----------\n==========\n")

        out.write('{}'.format(elapsed_time))
         
def main():
    in_dir = "../../res/instances"
    out_dir = "../out/rot/sol"
    sorted_ins = alphanumeric_sort(os.listdir(in_dir))
    timeout = 5
    
    for i in tqdm(range(len(sorted_ins))):
        trivial = Model(code)
        out = os.path.abspath(os.path.join(out_dir,f'out-{sorted_ins[i]}'))
        print(sorted_ins[i])
        
        w, n, cx, cy = read_ins(f'{in_dir}/{sorted_ins[i]}')
        instance = Instance(solver, trivial)
        
        
        min_height = max(max(cy), math.ceil(sum([cx[i] * cy[i] for i in range(n)]) / w))
        max_height = sum(cy)

        instance['n'] = n
        instance['cx'] = cx
        instance['cy'] = cy
        instance['w'] = w
        instance['lower'] = min_height
        instance['upper'] = max_height
        
        
        start_time = time.time()
        result = instance.solve(timeout=datetime.timedelta(minutes=timeout))
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        if elapsed_time > 300:
            print(f'Fail: Timeout!')

        else:
            start_x = result['x']
            start_y = result['y']
            height = result['h']
            rotate = result['is_rot']
            cxr = result['cxr']
            cyr = result['cyr']
        
            circuits = get_circuits(cxr, cyr, start_x, start_y)
            plot_solution(circuits, w, height, f'{sorted_ins[i]}', f'../out/rot/images/out-{sorted_ins[i]}.png', rotation = True, r = rotate)
            write_sol(w, n, cxr, cyr, start_x, start_y, height, out, elapsed_time)
            
if __name__ == '__main__':
    main()   
