import os
from tqdm import tqdm
from utils import *
import numpy as np
from z3 import *
import time
from glob import glob

def read_ins(ins):

    with open(ins, 'r') as in_file:

        l = in_file.read().splitlines()

        w = l[0]
        circ = l[1]

        cx = []
        cy = []

        for i in range(int(circ)):
            line = l[i + 2].split(' ')
            cx.append(int(line[0]))
            cy.append(int(line[1]))

        return int(w), int(circ), cx, cy

def write_sol(w, circ, cx, cy, x_sol, y_sol, h, sol, elapsed_time):

    with open(sol, 'w+') as out:

        out.write('{} {}\n'.format(w, h))
        out.write('{}\n'.format(circ))

        for i in range(circ):
            out.write('{} {} {} {}\n'.format(cx[i], cy[i], x_sol[i], y_sol[i]))
        
        out.write("----------\n==========\n")

        out.write('{}'.format(elapsed_time))


def main():
    in_dir = "../../res/instances"
    out_dir = "../out/rot/sol"
    sorted_ins = alphanumeric_sort(os.listdir(in_dir))
    for k in tqdm(range(len(sorted_ins))):

        out = os.path.abspath(os.path.join(out_dir,f'out-{sorted_ins[k]}'))
        
        w, circ, cx, cy = read_ins(f'{in_dir}/{sorted_ins[k]}')

        # Coordinates of the points
        x = IntVector('x',circ)  
        y = IntVector('y',circ)

        # Actual dimensions
        rot = BoolVector('rot', circ)
        cx_r = IntVector('cx_r', circ)
        cy_r = IntVector('cy_r', circ)
        
        # Maximum plate h to minimize
        lowest = max([min(cx[i], cy[i]) for i in range(circ)])
        lower = max(lowest, math.ceil(sum([cx[i] * cy[i] for i in range(circ)]) / w))
        upper = int(sum([max(cy[i], cx[i]) for i in range(circ)]))
        h = lower
        sol = False

        if not sol and h <= upper:
            # Setting the optimizer
            opt = Solver()
            opt.add(rot)

            # Setting domain and no overlap constraints
            domain_x = []
            domain_y = []
            no_overlap = []

            for i in range(circ):
                domain_x.append(x[i] >= 0)
                domain_x.append(x[i] + cx_r[i] <= w)
                domain_y.append(y[i]>=0)
                domain_y.append(y[i] + cy_r[i] <= h)
                

            
                for j in range(i+1, circ):
                    no_overlap.append(Or(x[i]+cx_r[i] <= x[j], x[j]+cx_r[j] <= x[i], y[i]+cy_r[i] <= y[j], y[j]+cy_r[j] <= y[i]))

                # If a circuit is squared, then force it to be not rotated
                opt.add(If(cx[i]==cy[i], And(cx[i]==cx_r[i], cy[i]==cy_r[i]), Or(And(cx[i]==cx_r[i], cy[i]==cy_r[i]), And(cx_r[i]==cy[i], cy_r[i]==cx[i]))))
                     
            opt.add(domain_x + domain_y + no_overlap)

            # Maximum time of execution
            opt.set("timeout", 300000)

            x_sol = []
            y_sol = []

            cx_sol = []
            cy_sol = []
            
            r = []

            # Solve

            print(f'{out}:', end='\t', flush=True)
            start_time = time.time()

            if opt.check() == sat:
                model = opt.model()
                elapsed_time = time.time() - start_time
                print(f'{elapsed_time * 1000:.1f} ms')
                # Getting values of variables
                for i in range(circ):
                    r.append(model.evaluate(rot[i], True))
                    x_sol.append(model.evaluate(x[i]).as_long())
                    y_sol.append(model.evaluate(y[i]).as_long())
                    cx_sol.append(model.evaluate(cx_r[i]).as_long())
                    cy_sol.append(model.evaluate(cy_r[i]).as_long())
                    if cx_sol[i] == cx[i]:
                        r[i] = False
                    elif cx_sol[i] == cy[i]:
                        r[i] = True

                # Storing the result
                circuits = get_circuits(cx, cy, x_sol, y_sol)
                plot_solution(circuits, w, h, f'{sorted_ins[k]}', f'../out/rot/images/out-{sorted_ins[k]}.png', rotation = True, r=r)
                write_sol(w, circ, cx_sol, cy_sol, x_sol, y_sol, h, out, elapsed_time)
                sol = True
            
            else:
                elapsed_time = time.time() - start_time
                print(f'{elapsed_time * 1000:.1f} ms')
                print("sol not found")
                
            h = h + 1
            
if __name__ == '__main__':
    main()   
