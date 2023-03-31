import os
from tqdm import tqdm
from utils import *
import numpy as np
import time
from glob import glob
from typing import List, Union
import math
import pulp

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
    out_dir = "../out/norot/sol"
    sorted_ins = alphanumeric_sort(os.listdir(in_dir))
    for k in tqdm(range(len(sorted_ins))):
        print(sorted_ins[k])
        out = os.path.abspath(os.path.join(out_dir,f'out-{sorted_ins[k]}'))

        W, n, cx, cy = read_ins(f'{in_dir}/{sorted_ins[k]}')
        
        
        prob = pulp.LpProblem("vlsi", pulp.LpMinimize)
        # Lower and upper bounds for the height
        lower = max(max(cy), math.ceil(sum([cx[i] * cy[i] for i in range(n)]) / W))
        upper = int(sum(cy))

        # Height variable
        l = pulp.LpVariable("l", lowBound=lower, upBound=upper, cat=pulp.LpInteger)
        prob += l, "Height of the plate"

        # Coordinate variables
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=int(W - cx[i]), cat=pulp.LpInteger)
                   for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=int(upper - cy[i]), cat=pulp.LpInteger)
                   for i in range(n)]

        # Boundary constraints
        for i in range(n):
            prob += x[i] + cx[i] <= W, f"X-axis of {i}-th coordinate bound"
            prob += y[i] + cy[i] <= l, f"Y-axis of {i}-th coordinate bound"

        # Booleans for OR condition
        set_C = range(2)
        delta = pulp.LpVariable.dicts(
            "delta",
            indices=(range(n), range(n), set_C),
            cat=pulp.LpBinary,
            lowBound=0,
            upBound=1,
        )

        # Non-Overlap constraints, at least one needs to be satisfied
        for i in range(n):
            for j in range(n):
                if i < j:
                    if cx[i] + cx[j] > W:
                        prob += delta[i][j][0] == 1
                        prob += delta[j][i][0] == 1

                    prob += x[i] + cx[i] <= x[j] + (delta[i][j][0]) * W
                    prob += x[j] + cx[j] <= x[i] + (delta[j][i][0]) * W
                    prob += y[i] + cy[i] <= y[j] + (delta[i][j][1]) * upper
                    prob += y[j] + cy[j] <= y[i] + (delta[j][i][1]) * upper
                    prob += (
                        delta[i][j][0] + delta[j][i][0] + delta[i][j][1] + delta[j][i][1]
                        <= 3
                    )

        # Solve

        # Get solution
        x_sol = []
        y_sol = []
        

        print(f'{out}:', end='\t', flush=True)
        start_time = time.time()

        prob.solve(solver=pulp.PULP_CBC_CMD(msg=True, timeLimit=300));              
        
        elapsed_time = time.time() - start_time
        
        if pulp.LpStatus[prob.status] == "Optimal" and elapsed_time <= 300:
            
            elapsed_time = time.time() - start_time
            print(f'{elapsed_time * 1000:.1f} ms')
            for i in range(n):
                x_sol.append(pulp.value(x[i]))
                y_sol.append(pulp.value(y[i]))
            h_sol = pulp.value(l)

            # Storing the result
            circuits = get_circuits(cx, cy, x_sol, y_sol)
            plot_solution(circuits, W, h_sol, f'{sorted_ins[k]}', f'../out/norot/images/out-{sorted_ins[k]}.png')
            write_sol(W, n, cx, cy, x_sol, y_sol, h_sol, out, elapsed_time)
            
        else:
            elapsed_time = time.time() - start_time
            print(f'{elapsed_time * 1000:.1f} ms')
            print("sol not found")
        
if __name__ == '__main__':
    main()   