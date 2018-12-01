# Physics 91SI
# Spring 2018
# AdmixMonte.py

# Import Mdodules
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

import numpy as np
from scipy import optimize
import pandas as pd
from cvxopt import matrix,solvers
from tabulate import tabulate

# First create functions that return an object of type "Data"
def use_data(filepath):
    df = pd.read_csv(filepath, index_col=0)
    return df


def getAdmix(Data, Target, Sources, print_results = True):

    try:
        # Takes in Target and Sources as lists of strings
        SourceDF = Data.loc[Sources]
        TargetDF = Data.loc[Target]

        # First make an empty results dataframe with columns labeled as Source populations
        results = pd.DataFrame(columns = SourceDF.index)
        # Add a single column for error values
        results['error'] = pd.Series()

        # Run quadratic program on each Target population
        for i, row in TargetDF.iterrows():
            # First keep track of the name of the target population we are calculating percentages for
            name = row.name

            # Then set up the quadratic problem.
            # Extract PCA coords of Target and Source populations
            Target = np.matrix(row.values).T
            Sources = np.matrix(SourceDF.values).T
            N = Sources.shape[1]
            nDims = Sources.shape[0]

            # Set up matrix A: Left side
            A0 = np.concatenate((Sources, np.ones((1,N))))
            A0 = np.concatenate((A0, -np.ones((1,N))))
            A0 = np.concatenate((A0, -np.identity(N)))
            # Set up matrix A: Right side
            A1 = np.identity(nDims)
            A1 = np.concatenate((A1, np.zeros((2, nDims))))
            A1 = np.concatenate((A1, np.zeros((N, nDims))))
            # Set up matrix A
            A = np.concatenate((A0, A1), axis = 1)

            # Set up vector B:
            B = np.concatenate((Target, [[1], [-1]]))
            B = np.concatenate((B, np.zeros((N, 1))))

            # Set up vector C:
            c = np.zeros((1, A.shape[1]))

            # Set up matrix H:
            H0 = np.zeros((N,N))
            H1 = np.zeros((N,nDims))
            H2 = H1.T
            H3 = np.identity((nDims))
            H_1 = np.concatenate((H0, H1), axis = 1)
            H_2 = np.concatenate((H2, H3), axis = 1)
            H = np.concatenate((H_1, H_2))

            # Run quadratic program
            x, obj = quadprog(A, B, c, H, print_message=False)
            result = np.array(x[:Sources.shape[1]].T)
            result = np.append(result, [-obj])
            result = pd.Series(result, name = name)
            result.index = results.columns

            results = results.append(result)

        # Print results
        if print_results == True:
            print(results.round(5))
        # Return a dataframe of admixture percentages
        return results
    except KeyError:
        print "One of your populations is not on file!"


def quadprog(A, b, c, H, print_message=True):
    '''
    arguments
    A: mxn np.matrix
    b: mx1 np.matrix
    c: 1xn np.matrix
    H: nxn matrix
    print_message: whether to print a message pertaining to success/failure of the code

    effects
    solves optimization problem of the form: max c x - x.T H x s.t. Ax <= b

    returns
    x: optimal solution as an nx1 np.matrix
    objective_value: optimal objective value
    '''
    solvers.options['show_progress'] = False
    solvers.options['feastol']=2e-7
    sol = solvers.qp(P=matrix(2.0*H), q=matrix(-1.0*c.T), G=matrix(1.0*A), h=matrix(1.0*b))
    if print_message:
        print sol['status']
    x = np.matrix(sol['x'])
    objective_value = -sol['primal objective']
    return x, objective_value



#############################################
# The rest of the file is already implemented
#############################################
"""
def run_dynamics(n, dt, xlim=(0, 1), ylim=(0, 1)):

    mol = init_molecule()

    # Animation stuff
    fig, ax = plt.subplots()
    line, = ax.plot((mol.p1.pos[0], mol.p2.pos[0]), (mol.p1.pos[1], mol.p2.pos[1]), '-o')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title('Dynamics simulation')
    dynamic_ani = animation.FuncAnimation(fig, update_anim, n,
            fargs=(dt, mol,line), interval=50, blit=False)
    plt.show()

def update_anim(i,dt, mol,line):
    pass

if __name__ == '__main__':
    # Set the number of iterations and time step size
    n = 10
    dt = .1
    run_dynamics(n, dt)
"""