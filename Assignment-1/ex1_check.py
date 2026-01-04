import time

import ex1
import search



def run_problem(func, targs=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    result = (-3, "default")
    try:
        result = func(*targs, **kwargs)

    except Exception as e:
        result = (-3, e)
    return result


# check_problem: problem, search_method, timeout
# timeout_exec: search_method, targs=[problem], timeout_duration=timeout
def solve_problems(problem, algorithm):
    

    try:
        p = ex1.create_watering_problem(problem)
    except Exception as e:
        print("Error creating problem: ", e)
        return None

    if algorithm == "gbfs":
        result = run_problem((lambda p: search.greedy_best_first_graph_search(p, p.h_gbfs)),targs=[p])
    else:
        result = run_problem((lambda p: search.astar_search(p, p.h_astar)), targs=[p])

    if result and isinstance(result[0], search.Node):
        solve = result[0].path()[::-1]
        solution = [pi.action for pi in solve][1:]
        print(len(solution), solution)
    else:
        print("no solution")



#Optimal : 20
Problem_pdf = {
    "Size":   (3, 3),
    "Walls":  {(0, 1), (2, 1)},
    "Taps":   {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
}

# Format reminder:
# {
#   "Size":   (N, M),
#   "Walls":  {(r,c), ...},
#   "Taps":   {(r,c): remaining_water, ...},
#   "Plants": {(r,c): required_water, ...},
#   "Robots": {rid: (r, c, load, capacity), ...}
# }

# -------------------------
# Problem 1: Tiny, no walls
# One robot, one tap, one plant
# -------------------------
#Optimal : 8
problem1 = {
    "Size":   (3, 3),
    "Walls":  set(),
    "Taps":   {(1, 1): 3},        # center
    "Plants": {(0, 2): 2},        # top-right
    "Robots": {
        10: (2, 0, 0, 2),         # bottom-left, cap 2
    },
}

# -------------------------
# Problem 2: Small with walls (your example-style)
# Two robots, one tap, two plants, vertical walls
# -------------------------
#Optimal: 20
problem2 = {
    "Size":  (3, 3),
    "Walls": {(0, 1), (2, 1)},    # middle column walls in top & bottom rows
    "Taps":  {(1, 1): 6},         # center
    "Plants": {
        (0, 2): 3,                # top-right
        (2, 0): 2,                # bottom-left
    },
    "Robots": {
        10: (1, 0, 0, 2),         # middle-left
        11: (1, 2, 0, 2),         # middle-right
    },
}




# -------------------------
# Problem 3: Corridor with walls, 5x3, one robot shuttling
# -------------------------
#optimal: 28
problem3 = {
    "Size":  (5, 3),              # rows: 0..4, cols: 0..2
    "Walls": {(1, 1), (3, 1)},    # walls in the middle column
    "Taps": {
        (0, 0): 5,                # top-left
    },
    "Plants": {
        (4, 2): 4,                # bottom-right
    },
    "Robots": {
        10: (2, 0, 0, 2),         # middle-left, cap 2 → needs multiple trips
    },
}

# -------------------------
# Problem 4
# -------------------------
#optimal: 13
problem4 = {
    "Size":  (5, 5),
    "Walls": {(0, 1),(1, 1),(2, 1), (0, 3),(1, 3),(2, 3)},    # two blocked cells
    "Taps": {
        (3, 2): 1,                # top-left
        (4, 2): 1,                # bottom-right
    },
    "Plants": {
        (0, 2): 1,                # top-right
        (1, 2): 1,                # bottom-left
                        # somewhere in middle-left
    },
    "Robots": {
        10: (3, 1, 0, 1),         # near left side
        11: (3, 3, 0, 1),         # near right side
    },
}

# -------------------------
# Problem 5: Intentional dead-end (not enough water)
# Good to test your dead-end pruning
# -------------------------
problem5_deadend = {
    "Size":  (3, 4),
    "Walls": set(),
    "Taps": {
        (1, 1): 3,                # only 3 units in world
    },
    "Plants": {
        (0, 3): 2,
        (2, 3): 2,                # total need = 4 > 3 → impossible
    },
    "Robots": {
        10: (1, 0, 0, 2),
    },
}
# -------------------------
# Problem 6:
# -------------------------
#optimal: 8
problem6 = {
    "Size":  (8, 8),
    "Walls": {
        # All cells except the corridor (1,0), (1,1), (1,2)
        *( (r, c)
           for r in range(8)
           for c in range(8)
           if not (r == 1 and c in (0, 1, 2)) )
    },
    "Taps": {
        (1, 1): 3,
    },
    "Plants": {
        (1, 2): 3,
    },
    "Robots": {
        10: (1, 0, 0, 3),   # start left of tap, cap 3
    },
}
#optimal: 20
problem7 = {
    "Size":  (4, 4),

    "Walls": set(),  # everything open

    "Taps": {
        (2, 2): 18,       # center tap
    },

    "Plants": {
        (0, 3): 3,        # top-right
        (3, 0): 3,        # bottom-left
        # total need = 8, tap has 18 (some slack)
    },

    "Robots": {
        10: (2, 1, 0, 3),  # left of tap, capacity 3
        11: (2, 0, 0, 3),  # right of tap, capacity 3
    },
}

#optimal: 31
problem_hard1 = {
    "Size":  (5, 6),

    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2), (1, 3),
        (3, 2), (3, 3),
    },

    "Taps": {
        (2, 2): 12,
    },

    "Plants": {
        (0, 1): 3,
        (4, 5): 6,
    },

    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
}

#optimal: 24
problem_hard2 = {
    "Size":  (5, 5),

    # Open plus-shape in center, corners blocked
    "Walls": {
        (0, 0), (0, 4),
        (4, 0), (4, 4),
    },

    "Taps": {
        (2, 2): 9,       # center
    },

    "Plants": {
        (1, 2): 3,       # up
        (2, 1): 3,       # left
        (3, 2): 3,       # down
    },

    "Robots": {
        10: (2, 3, 0, 3),  # right of tap
    },
}
#optimal: 42
problem_hard3 = {
    "Size":  (5, 5),

    # Open plus-shape in center, corners blocked
    "Walls": {
        (0, 0), (0, 4),
        (4, 0), (4, 4),
    },

    "Taps": {
        (2, 2): 9,       # center
    },

    "Plants": {
        (1, 2): 3,       # up
        (2, 1): 3,       # left
        (3, 2): 3,       # down
    },

    "Robots": {
        10: (2, 3, 0, 3),  # right of tap
    },
}

#optimal: 25
problem_hard4 = {
    "Size":  (5, 6),

    "Walls": {
        # corridor shifted up
        (0, 2), (0, 3),
        (2, 2), (2, 3),
    },

    "Taps": {
        (1, 2): 6,         # upper tap
        (3, 3): 6,         # lower tap
    },

    "Plants": {
        (0, 0): 3,         # top-left
        (4, 5): 3,         # bottom-right
    },

    "Robots": {
        10: (1, 1, 0, 3),  # near upper tap, cap 3
        11: (3, 4, 0, 2),  # near lower tap, cap 2
    },
}

#optimal: 29
problem_hard5 = {
    "Size":  (5, 6),

    "Walls": {
        # corridor shifted up
        (0, 2), (0, 3),
        (2, 2), (2, 3),
    },

    "Taps": {
        (1, 2): 8,         # upper tap
        (3, 3): 8,         # lower tap
    },

    "Plants": {
        (0, 0): 4,         # top-left
        (4, 5): 4,         # bottom-right
    },

    "Robots": {
        10: (1, 1, 0, 4),  # near upper tap, cap 3
        11: (3, 4, 0, 3),  # near lower tap, cap 2
    },
}

#optimal: 33
problem_hard6 = {
    "Size":  (5, 6),

    "Walls": {
        # corridor shifted up
        (0, 2), (0, 3),
        (2, 2), (2, 3),
    },

    "Taps": {
        (1, 2): 10,         # upper tap
        (3, 3): 10,         # lower tap
    },

    "Plants": {
        (0, 0): 5,         # top-left
        (4, 5): 5,         # bottom-right
    },

    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
}

#optimal 65
problem_load = {
    "Size":  (10, 4),
    "Walls": {
        (0,1),(1, 1), (2, 1), (3, 1), (4, 1), (6, 1),
        (7, 1), (8, 1), (9, 1),(4,2), (4,3),(6,2), (6,3)
    },

    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },

    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,    # upper-right corrido
        (9, 0): 10,   
    },

    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),   # bottom-left area near the tap side
        11: (7, 0, 0, 20),   # bottom-left area near the tap side
    },
}

#optimal: 106
problem_10x10_single = {
    "Size":  (10, 10),

    "Walls": set(),   # completely open grid

    "Taps": {
        (5, 5): 24,   # central tap
    },

    "Plants": {
        (0, 0): 5,    # top-left
        (0, 9): 5,    # top-right
        (9, 0): 5,    # bottom-left
        (9, 9): 5,    # bottom-right
        # total need = 20
    },

    "Robots": {
        10: (9, 5, 0, 5),   # bottom-middle, capacity 3
    },
}
#optimal: 249
problem_12x12_snake = {
    "Size":  (12, 12),

    # Three vertical "barriers" with single / few gaps:
    # - Column 3 blocked except row 5 (door there).
    # - Column 6 blocked except rows 2 and 9 (two doors).
    # - Column 9 blocked except row 7 (door).
    "Walls": {
        (1, 3), (2, 3), (3, 3), (4, 3),
        (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),

        (1, 6), (3, 6), (4, 6), (5, 6),
        (6, 6), (7, 6), (8, 6), (10, 6),

        (1, 9), (2, 9), (3, 9), (4, 9),
        (5, 9), (6, 9), (8, 9), (9, 9), (10, 9),
    },

    # Tap on the left side, with enough water
    "Taps": {
        (5, 1): 24,
    },

    # Plants on the far right, all need water
    "Plants": {
        (2, 11): 5,    # upper-right corridor
        (7, 11): 5,    # mid-right
        (10, 10): 5,   # lower-right
        (0, 8): 5,     # top, near the last barrier
        # total need = 20
    },

    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (11, 1, 0, 3),   # bottom-left area near the tap side
    },
}

#optimal: 343
problem_12x12_snake_hard = {
    "Size":  (12, 12),

    # Three vertical "barriers" with single / few gaps:
    # - Column 3 blocked except row 5 (door there).
    # - Column 6 blocked except rows 2 and 9 (two doors).
    # - Column 9 blocked except row 7 (door).
    "Walls": {
        (1, 3), (2, 3), (3, 3), (4, 3),
        (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),

        (1, 6), (3, 6), (4, 6), (5, 6),
        (6, 6), (7, 6), (8, 6), (10, 6),

        (1, 9), (2, 9), (3, 9), (4, 9),
        (5, 9), (6, 9), (8, 9), (9, 9), (10, 9),
    },

    # Tap on the left side, with enough water
    "Taps": {
        (5, 1): 24,
    },

    # Plants on the far right, all need water
    "Plants": {
        (2, 11): 5,    # upper-right corridor
        (7, 11): 5,    # mid-right
        (10, 10): 5,   # lower-right
        (0, 8): 5,     # top, near the last barrier
        # total need = 20
    },

    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (11, 1, 0, 2),   # bottom-left area near the tap side
    },
}

# 37
problem_hard7 = {
    "Size":  (5, 6),
    "Walls": {(0, 2), (0, 3), (2, 2), (2, 3)},
    "Taps": {(1, 2): 10, (3, 3): 10},
    "Plants": {(0, 0): 6, (4, 5): 6},
    "Robots": {10: (1, 1, 0, 6), 11: (3, 4, 0, 5)},
}

# 52
problem_harder44 = {
    "Size": (10, 6),
    "Walls": {
        (1, 2), (1, 3),
        (2, 2), (2, 3),
        (4, 1), (4, 2), (4, 3), (4, 4),
        (6, 2), (6, 3),
        (7, 2), (7, 3),
        (8, 4), (8, 5),
    },
    "Taps": {(0, 3): 10, (9, 2): 10},
    "Plants": {(0, 0): 2, (9, 5): 2, (5, 5): 1},
    "Robots": {10: (2, 0, 0, 1), 11: (7, 5, 0, 1)},
}

# optimal: 13
problem4 = {
    "Size":  (5, 5),
    "Walls": {(0, 1),(1, 1),(2, 1), (0, 3),(1, 3),(2, 3)},
    "Taps": {(3, 2): 1, (4, 2): 1},
    "Plants": {(0, 2): 1, (1, 2): 1},
    "Robots": {10: (3, 1, 0, 1), 11: (3, 3, 0, 1)},
}

def main():
    start = time.time()
    problem = [ problem4,
                problem_hard7,
                problem_harder44,
                problem_10x10_single,
                problem_12x12_snake,
                problem_12x12_snake_hard,
                problem_hard1,
                problem_hard2,
                problem_hard3,
                problem_hard4,
                problem_hard5,
                problem_hard6,
                problem5_deadend
                ]
    for p in problem:
        for a in ['astar']:
            solve_problems(p, a)
    end = time.time()
    print('Submission took:', end-start, 'seconds.')


if __name__ == '__main__':
    main()
