
# CP

##Requirements

pip install minizinc

pip install matplotlib

pip install tqdm

##Run from Command Prompt

Please run the project from the src folder 'Jabbour\CP\src'

To run the normal model:

python cp

To run the rotated model:

python cp_rot

##Note

I have left the folder 'rot' and 'no-rot' empty so that my outputs would not be overwritten when the models are run again. My outputs refere to the runs that were mentioned in the paper and are seen in Tables 1 and 2.

To be able to run the experiments that I mentioned in the paper, i provided a user input to choose the solver in my normal model, gecode or chuffed. Furthermore, you can recreate the models without symmetry breaking constraints by opening the minizinc scripts, no_rot.mzn and rot.mzn, and commenting out the section with the symmetry breaking constraints. Finally for the restarts, I kept them commented out in the aforementioned minizinc scripts so that you can remove the comments and run them if need be; note that if that is to be done you have to switch the search at the end, run the one I have commented out and comment out the one I have running.

I have added comments in the minizinc scripts that indicate what I explained above.

