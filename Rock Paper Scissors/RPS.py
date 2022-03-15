# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago.
# It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np
import random

def player(prev_play, our_prev_play, q_table, epsilon,win):

    # initilizations
    action_space = [0,1,2]
    epsilon_decay = 0.005

    state = state_decoder(prev_play, our_prev_play)

    reward = win

    guess = epsilon_greedy_action_selection(epsilon, q_table, state, action_space)

    new_state = state_decoder(prev_play, guess)

    old_q_value = q_table[state,guess]

    next_optimal_q_value = np.max(q_table[new_state,:])

    next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)

    q_table[state,guess] = next_q

    epsilon = decay_epsilon(epsilon,epsilon_decay)

    if guess == 0:

        guess = "R"

    elif guess == 1:

        guess = "P"

    elif guess == 2:

        guess = "S"

    return guess, q_table, epsilon


def epsilon_greedy_action_selection(epsilon, q_table, state, actions):

    random_number = np.random.random()

    if random_number > epsilon:

        state_row = q_table[state,:]

        action = np.argmax(state_row)

    else:

        action = random.choice(actions)

    return action


def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    alpha = 0.8

    return old_q_value + alpha * (reward + next_optimal_q_value - old_q_value)


def decay_epsilon(epsilon, epsilon_decay):

    if epsilon <= 0.1:

        return epsilon

    else:

        return epsilon - epsilon_decay


def state_decoder(prev_play, our_prev_play):

    if (prev_play == "R" or prev_play == 0) and (our_prev_play == "R" or our_prev_play == 0):

        return 0

    elif (prev_play == "R" or prev_play == 0) and (our_prev_play == "P" or our_prev_play == 1):

        return 1

    elif (prev_play == "R" or prev_play == 0) and (our_prev_play == "S" or our_prev_play == 2):

        return 2

    elif (prev_play == "P" or prev_play == 1) and (our_prev_play == "R" or our_prev_play == 0):

        return 3
    elif (prev_play == "P" or prev_play == 1) and (our_prev_play == "P" or our_prev_play == 1):

        return 4
    elif (prev_play == "P" or prev_play == 1) and (our_prev_play == "S" or our_prev_play == 2):

        return 5
    elif (prev_play == "S" or prev_play == 2) and (our_prev_play == "R" or our_prev_play == 0):

        return 6
    elif (prev_play == "S" or prev_play == 2) and (our_prev_play == "P" or our_prev_play == 1):

        return 7
    elif (prev_play == "S" or prev_play == 2) and (our_prev_play == "S" or our_prev_play == 2):

        return 8




