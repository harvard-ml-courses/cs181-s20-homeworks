import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import gridworld

# Change to False once part one (print_grid_representations) is done.
run_part_one = True

# ----------------------------------------------------------------------- #
#   Starter code for CS 181 2020 HW 6, Problem 2                          #
# ----------------------------------------------------------------------- #
#
# Please read all of T6_P2.py before beginning to code.

#############################################################
#          DO NOT MODIFY THIS REGION OF THE CODE.           #

np.random.seed(181)
VALUE_ITER = 'Value'
POLICY_ITER = 'Policy'

### HELPER CODE ####

### INIITALIZE GRID ###

# Create the grid for Problem 2.
grid = [
    '..,..',
    '..,..',
    'o.,..',
    '.?,.*']

# Create the Task
# Task Parameters
action_error_prob = .2

task = gridworld.GridWorld(grid,
                            action_error_prob=action_error_prob,
                            terminal_markers={'*', '?'},
                            rewards={'.': -1, '*': 50, '?': 5, ',': -50, 'o': -1} )

# Algorithm Parameters
gamma = .75
state_count = task.num_states
action_count = task.num_actions
row_count = len( grid )
col_count = len( grid[0] )

# -------------- #
#   Make Plots   #
# -------------- #

# Util to make an arrow
# The directions are [ 'north' , 'south' , 'east' , 'west' ]
def plot_arrow( location , direction , plot ):
    arrow = plt.arrow( location[0] , location[1] , dx , dy , fc="k", ec="k", head_width=0.05, head_length=0.1 )
    plot.add_patch(arrow)

# Util to make a value function plot from the current Q_table
def make_value_plot(V, pi):
    # Useful stats for the plot
    value_function = np.reshape( V , ( row_count , col_count ) )
    policy_function = np.reshape( pi , ( row_count , col_count ) )

    # Write the value on top of each square
    indx, indy = np.arange(row_count), np.arange(col_count)
    x, y = np.meshgrid(indx, indy)

    fig, ax = plt.subplots()
    ax.imshow( value_function , interpolation='none' , cmap= plt.get_cmap('Reds_r'))

    s = 0
    for s in range(len(V)):
        val = V[s]
        (xval, yval) = task.maze.unflatten_index(s)
        t = "%.2f"%(val,) # format value with 1 decimal point

        ax.text(yval, xval, t, color='black', va='center', ha='center', size=15)


# Util to make a policy plot from the current Q_table
def make_policy_plot(V, pi, iter_type = VALUE_ITER, iter_num = 0):
    # Useful stats for the plot
    row_count = len( grid )
    col_count = len( grid[0] )
    value_function = np.reshape( V , ( row_count , col_count ) )
    policy_function = np.reshape( pi , ( row_count , col_count ) )

    for row in range( row_count ):
        for col in range( col_count ):
            if policy_function[row,col] == 0:
                dx = 0; dy = -.5
            if policy_function[row,col] == 1:
                dx = 0; dy = .5
            if policy_function[row,col] == 2:
                dx = .5; dy = 0
            if policy_function[row,col] == 3:
                dx = -.5; dy = 0
            plt.arrow( col , row , dx , dy , shape='full', fc='w' , ec='gray' , lw=1., length_includes_head=True, head_width=.1 )
    plt.title( iter_type + ' Iteration, i = ' + str(iter_num) )
    plt.savefig(iter_type + '_' + str(iter_num) + '.png')
    plt.show( block=True)
    # Save the plot to the local directory.

#############################################################
#          HELPER FUNCTIONS - DO NOT MODIFY                 #

def unflatten_index(flat_i):
    """
    flat_i is an flattened state index.
    Returns the unflattened representation of the state at flat_i.
    """
    return task.maze.unflatten_index(flat_i)

def flatten_index(unflat_i):
    """
    unflat_i is an unflattened state index.
    Returns the flattened representation of the state at unflat_i.
    """
    return task.maze.flatten_index(unflat_i)

def get_reward(state, action):
    """
    state is a flattened state.
    action represents an index into the actions array.

    Returns the reward from exiting the state.
    The reward only depends on the given state that the agent is leaving.
    """
    return task.rewards.get(task.maze.get_flat(state))

def is_wall(state):
    """
    state represents a flattened state.
    Returns true if state is a wall.
    """
    if state < 0 or state > state_count:
        return True
    return task.is_wall(task.maze.unflatten_index(state))

def get_transition_prob(state1, action1, state2):
    """
    state1 and state2 are flattened states.
    action1 represents an index into the actions array.

    Returns p(state2 | state1, action1).
    """
    return task.get_transition_prob(state1, action1, state2)

#############################################################
#          TO-DOS FOR PROBLEM 2                             #

def print_grid_representations():
    """
    Please complete the tasks in this function.
    Do not call any functions in gridworld.py.
    Your solution should only call helper functions in T6_P3.py.
    """

    # In Gridworld, each state on the grid can be represented using an
    # unflattened or a flattened index.
    # The unflattened index is a tuple (x, y) representing the state's position
    # on the grid.
    # The flattened index is a single integer.  Each integer in range(state_count)
    # corresponds to a particular state.

    # Here is some code to convert unflattened indices to flattened indices.
    # Unflattened indices -> flattened indices.
    for r_ind in range(row_count):
        for c_ind in range(col_count):
            u_ind = (r_ind, c_ind)
            print(str(u_ind) + ' is converted to ' + str(flatten_index(u_ind)))

    # Flattened indices -> unflattened indices.
    for s in range(state_count):
        # TODO 1: Write code to convert flattened index s to an unflattened index s_u.
        # You can check your answer by comparing your print statements to those above.
        # Your implementation must call a helper function in this file.
        s_u = (0, 0)
        print(str(s) + ' is converted to ' + str(s_u))

    # Recall that when you take an action in Gridworld, you won't always
    # necessarily move in that direction.  Instead there is some probability of
    # moving to a state on either side.

    starting_state = 14
    for a in range(action_count):
        for new_state in range(state_count):
            # TODO 2: Change the condition in this if statement
            # to only print when new_state has nonzero probability
            # p(new state | starting_state, a).
            # Your implementation must call a helper function in this file.
            prb = 0.
            if prb > 0:
                print('Going from ' + str(unflatten_index(starting_state)) + ' to '
                    + str(unflatten_index(new_state)) + ' when taking action '
                    + str(task.actions[a]) + ' has probability ' + str(prb))

def policy_evaluation(pi, gamma, theta = 0.1):
    """
    Returns array V containing the policy evaluation of policy pi.
    Implement policy evaluation using discount factor gamma and
    convergence tolerance theta.

    In your implementation, please use helper functions get_transition_prob
    and get_reward in this file.  You do not need to calculate the transition
    probabilities yourself.
    """
    # TODO: Complete this function.
    V = np.zeros(state_count)
    return V

def update_policy_iteration(V, pi, gamma, theta = 0.1):
    """
    Return updated V_new and pi_new using policy iteration.
    V represents the learned value function at each state.
    pi represents the learned policy at each state.

    In your implementation, please use helper functions get_transition_prob
    and get_reward in this file.  You do not need to calculate the transition
    probabilities yourself.
    """
    # TODO: Complete this function.
    V_new = policy_evaluation(pi, gamma, theta)
    pi_new = np.zeros(state_count)

    return V_new, pi_new

def update_value_iteration(V, pi, gamma):
    """
    Return updated V_new and pi_new using value iteration.
    V represents the learned value function at each state.
    pi represents the learned policy at each state.

    In your implementation, please use helper functions get_transition_prob
    and get_reward in this file.  You do not need to calculate the transition
    probabilities yourself.
    """
    # TODO: Complete this function.
    V_new = np.zeros(state_count)
    pi_new = np.zeros(state_count)

    return V_new, pi_new

"""
Do not modify the learn_strategy method, but please read through its code.
"""
def learn_strategy(planning_type= VALUE_ITER, max_iter = 10, print_every = 5, ct = None):
    # Loop over some number of episodes
    n_iter = 0
    V = np.zeros( state_count )
    pi = np.zeros( state_count )

    while n_iter < max_iter:
        # Initialize the Q table
        if n_iter >= max_iter:
            break

        while True:
            n_iter += 1
            V_prev = V.copy()

            # update V and pi
            if planning_type == VALUE_ITER:
                V, pi = update_value_iteration(V, pi, gamma)

                if (n_iter % print_every == 0):
                    # make value plot
                    make_value_plot(V = V, pi = pi)
                    # plot the policy
                    make_policy_plot(V = V, pi = pi, iter_type = VALUE_ITER, iter_num = n_iter)

                if ct:
                    # calculate the difference between this V and the previous V
                    diff = np.absolute(np.subtract(V, V_prev))
                    # check that every component is less than ct
                    i = 0
                    while(i < state_count and diff[i] < ct):
                        i += 1
                    if i == state_count:
                        print("Converged at iteration " + str(n_iter))
                        # make value plot
                        make_value_plot(V = V, pi = pi)
                        # plot the policy
                        make_policy_plot(V = V, pi = pi, iter_type = VALUE_ITER, iter_num = n_iter)
                        return 0

            elif planning_type == POLICY_ITER:
                V, pi = update_policy_iteration(V, pi, gamma, theta = 0.1)
                if (n_iter % print_every == 0):
                    # make value plot
                    make_value_plot(V = V, pi = pi)
                    # plot the policy
                    make_policy_plot(V = V, pi = pi, iter_type = POLICY_ITER, iter_num = n_iter)

            if n_iter >= max_iter:
                break

if run_part_one:
    print_grid_representations()

else:
    print('Beginning policy iteration.')
    learn_strategy(planning_type=POLICY_ITER, max_iter = 10, print_every = 2)
    print('Policy iteration complete.')

    print('Beginning value iteration.')
    learn_strategy(planning_type=VALUE_ITER, max_iter = 10, print_every = 2)
    print('Value iteration complete.\n')
