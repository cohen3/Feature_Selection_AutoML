import math
import string

import numpy as np
import numpy.random as rn
import random as r
from scipy import optimize
import re
from collections import Counter

WORD = re.compile(r"\w+")

chars = [c for c in string.ascii_lowercase]
interval = (-10, 10)


def annealing(random_start,
              cost_function,
              random_neighbour,
              acceptance,
              temperature,
              maxsteps=1000,
              max_rejects=0,
              debug=True):
    """ Optimize the black-box function 'cost_function' with the simulated annealing algorithm."""
    state = random_start()
    cost, real_f1 = cost_function(state)
    states, costs, real_scores = [state], [cost], []
    rejects_counter = 0
    for step in range(maxsteps):
        fraction = step / float(maxsteps)
        T = temperature(fraction)
        new_state = random_neighbour(state, fraction)
        new_cost, real_f1 = cost_function(new_state)
        if real_f1 is not None:
            real_scores.append(real_f1)
        if debug:
            print("Step #{:>2}/{:>2} : T = {:>4.3g}, state = {}, cost = {:>4.3g}, new_state = {}, "
                  "new_cost = {:>4.3g} ...".format(step, maxsteps, T, state, cost, new_state, new_cost))
        if acceptance(cost, new_cost, T) > rn.random():
            state, cost = new_state, new_cost
            states.append(state)
            costs.append(cost)
            rejects_counter = 0
        #     print("  ==> Accept it!")
        else:
            rejects_counter += 1
        #     print("  ==> Reject it...")
        if 0 < max_rejects == rejects_counter:
            return state, cost_function(state)[0], states, costs, real_scores
    print(cost_function(state)[0])
    return state, cost_function(state)[0], states, costs, real_scores


def f(x):
    """ Function to minimize."""

    sol = 'eladcohen'
    res = 0
    for c in x:
        if c in sol:
            sol = sol.replace(c, '', 1)
            res += 1
        else:
            res -= 1

    return res


def clip(x):
    """ Force x to be in the interval."""
    a, b = interval
    return max(min(x, b), a)


def random_start():
    """ Random point in the interval."""
    sample = list(rn.choice(chars, rn.randint(0, len(chars))))
    return sample


def cost_function(x):
    """ Cost of x = f(x)."""
    return f(x) * -1.0


def random_neighbour(x, fraction=1):
    """Move a little bit x, from the left or the right."""
    neighbours = list()
    for c in string.ascii_lowercase:
        neighbours.append(x + [c])

    for c in x:
        neighbours.append([x for x in x if x != c])
    return list(r.choice(neighbours))


def acceptance_probability(cost, new_cost, temperature):
    if new_cost < cost:
        # print("    - Acceptance probabilty = 1 as new_cost = {} < cost = {}...".format(new_cost, cost))
        return 1
    else:
        p = np.exp(- (new_cost - cost) / temperature)
        # print("    - Acceptance probabilty = {:.3g}...".format(p))
        return p


def temperature(fraction):
    """ Example of temperature dicreasing as the process goes on."""
    return max(0.01, min(1, 1 - fraction))


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


if __name__ == '__main__':
    state, c, states, costs, _ = annealing(random_start, cost_function, random_neighbour, acceptance_probability,
                                           temperature, maxsteps=1000, debug=True)
    print(sorted(state))
