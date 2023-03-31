import os
from tqdm import tqdm
from utils import *
import numpy as np
import time
from glob import glob
from typing import List, Union
import math
import pulp

def plot(circuits, width, height, title, file='', rot = False, r=[]):
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    fig.canvas.manager.set_window_title(title)

    for i,(w,h,x,y) in enumerate(circuits):
        if rot and r[i] == 1:
            w, h = h, w
        rect = patches.Rectangle((x, y), w, h, linewidth = 2, edgecolor= 'black', facecolor = colors.hsv_to_rgb((i / len(circuits), 1, 1)))
        ax.add_patch(rect)
        
    ax.set_yticks(np.arange(height+1))
    ax.set_xticks(np.arange(width+1))
    ax.grid(color='black', linewidth = 1)
    
    if file is not None:
        plt.savefig(file)
        plt.close()
    else:
        plt.show()

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
    for k in tqdm(range(len(sorted_ins))):

        out = os.path.abspath(os.path.join(out_dir,f'out-{sorted_ins[k]}'))
        W, n, cx, cy = read_ins(f'{in_dir}/{sorted_ins[k]}')
        
        
        range(n) = range(n)
        prob = pulp.LpProblem("vlsi_rot", pulp.LpMinimize)
        # Lower and upper bounds for the height
        lowest = max([min(cx[i], cy[i]) for i in range(n)])
        lower = max(lowest, math.ceil(sum([cx[i] * cy[i] for i in range(n)]) / W))
        upper = int(sum([max(cy[i], cx[i]) for i in range(n)]))

        # Height variable
        l = pulp.LpVariable("l", lowBound=lower, upBound=upper, cat=pulp.LpInteger)
        prob += l, "Height of the plate"

        # Coordinate variables
        x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=int(W - min(cx[i],cy[i])), cat=pulp.LpInteger)
                   for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", lowBound=0, upBound=int(upper - min(cx[i],cy[i])), cat=pulp.LpInteger)
                   for i in range(n)]
        
        # Rotation variables
        rotation = pulp.LpVariable.dicts(
            "rot", indices=range(n), lowBound=0, upBound=1, cat=pulp.LpBinary)

        for i in range(n):
            if cx[i] == cy[i] or cy[i] > W:
                rotation[i] = 0
                
        cxr = [pulp.LpVariable("cxr_{i}", lowBound=0) for i in range(n)]
        cyr = [pulp.LpVariable("cyr_{i}", lowBound=0) for i in range(n)]
        
        for i in range(n):
            if rotation[i] == 1:
                cxr[i] = cy[i]
                cyr[i] = cx[i]
            else:
                cxr[i] = cx[i]
                cyr[i] = cy[i]

        # Boundary constraints
        for i in range(n):
            prob += (x[i] + cx[i] * (1 - rotation[i]) + cy[i] * rotation[i] <= W, f"X-axis of {i}-th coordinate bound")
            prob += (y[i] + cy[i] * (1 - rotation[i]) + cx[i] * rotation[i] <= l, f"Y-axis of {i}-th coordinate bound")

        # Booleans for OR condition
        set_C = range(2)
        delta = pulp.LpVariable.dicts(
            "delta",
            indices=(range(n), range(n), set_C),
            cat=pulp.LpBinary,
            lowBound=0,
            upBound=1,
        )

        max_circuit = np.argmax(np.asarray(cx) * np.asarray(cy))
        prob += x[max_circuit] == 0, "Max circuit in x-0"
        prob += y[max_circuit] == 0, "Max circuit in y-0"

        # Non-Overlap constraints, at least one needs to be satisfied
        for i in range(n):
            for j in range(n):
                if i < j:
                    if all(
                        [
                            (u + v) > W
                            for u in [cx[i], cy[i]]
                            for v in [cx[j], cy[i]]
                        ]
                    ):
                        prob += delta[i][j][0] == 1
                        prob += delta[j][i][0] == 1

                    prob += (
                        x[i]
                        + cx[i] * (1 - rotation[i])
                        + cy[i] * rotation[i]
                        <= x[j] + delta[i][j][0] * W
                    )
                    prob += (
                        x[j]
                        + cx[j] * (1 - rotation[j])
                        + cy[j] * rotation[j]
                        <= x[i] + delta[j][i][0] * W
                    )

                    prob += (
                        y[i]
                        + cy[i] * (1 - rotation[i])
                        + cx[i] * rotation[i]
                        <= y[j] + delta[i][j][1] * upper
                    )
                    prob += (
                        y[j]
                        + cy[j] * (1 - rotation[j])
                        + cx[j] * rotation[j]
                        <= y[i] + delta[j][i][1] * upper
                    )

                    prob += (
                        delta[i][j][0] + delta[j][i][0] + delta[i][j][1] + delta[j][i][1]
                        <= 3
                    )


        # Solve

        # Get solution
        x_sol = []
        y_sol = []
        cx_sol = []
        cy_sol = []
        r = []

        print(f'{out}:', end='\t', flush=True)
        start_time = time.time()

        prob.solve(solver=pulp.PULP_CBC_CMD(msg=True, timeLimit=300))               
        
        elapsed_time = time.time() - start_time
        
        if pulp.LpStatus[prob.status] == "Optimal" and elapsed_time <= 300:
            
            elapsed_time = time.time() - start_time
            print(f'{elapsed_time * 1000:.1f} ms')
            for i in range(n):
                r.append(pulp.value(rotation[i]))
                x_sol.append(pulp.value(x[i]))
                y_sol.append(pulp.value(y[i]))
                cx_sol.append(pulp.value(cxr[i]))
                cy_sol.append(pulp.value(cyr[i]))
            h_sol = pulp.value(l)

            # Storing the result
            print('r', r) 
            circuits = get_circuits(cx, cy, x_sol, y_sol)
            plot(circuits, W, h_sol, f'{sorted_ins[k]}', f'../out/rot/images/out-{sorted_ins[k]}.png', rot = True, r=r)
            write_sol(W, n, cx_sol, cy_sol, x_sol, y_sol, h_sol, out, elapsed_time)
            
        else:
            elapsed_time = time.time() - start_time
            print(f'{elapsed_time * 1000:.1f} ms')
            print("sol not found")

if __name__ == '__main__':
    main()   
