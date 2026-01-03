from __future__ import generators

import math, random, sys, re, bisect, string
import operator, copy, os.path, inspect
import ext_plant

class Controller:
    def __init__(self, game: ext_plant.Game):
        self.game = game
        self.problem_desc = game.get_problem()
        self.goal_reward = game._goal_reward
        self.rows, self.cols = self.problem_desc["Size"]
        self.walls = set(self.problem_desc["Walls"])
        self.capacities = game.get_capacities()
        self.active_robots = {}
        self.target_plants = {}
        self.is_small_world = self.rows * self.cols <= 9
        self.extension_check = False # Fixed variable name used in choose_next_action

        # mean rewards calculation
        raw_rewards = self.problem_desc.get("plants_reward", {})
        self.mean_rewards = {
            pos: (sum(plants) / len(plants)) 
            for pos, plants in raw_rewards.items()
        }
        
        # calculating max probabilty
        probs = self.problem_desc.get("robot_chosen_action_prob", {})
        self.MAX_PROB = 0
        for p in probs.values():
            if p > self.MAX_PROB:
                self.MAX_PROB = p

        # distance helper for robot selection
        def get_real_dist(start, end):
            if start == end: return 0
            queue = [(start, 0)]
            visited = {start}
            while queue:
                (r, c), dist = queue.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) == end: return dist + 1
                    if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                        (nr, nc) not in self.walls and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
            return 10000

        # identify the "best" target using find_targets logic
        current_state = game.get_current_state()
        taps_dict = dict(current_state[2])
        best_plant_pos = None
        highest_target_score = -1

        for pos, need in current_state[1]:
            if need > 0:
                reward_val = self.mean_rewards.get(pos, 1)
                
                d2 = min([get_real_dist(pos, tap) for tap in taps_dict.keys()] + [999])

                p_score = (reward_val ** 2) / (d2 + 1)
                if p_score > highest_target_score:
                    highest_target_score = p_score
                    best_plant_pos = pos

        # primary robot selection based on probability, capacity, and distance to the identified best plant
        robot_initial_positions = {r_id: pos for r_id, pos, _ in current_state[0]}
        
        # robot score calculation
        def robot_selection_score(rid):
            p = probs.get(rid, 0)
            c = self.capacities.get(rid, 0)
            d1 = get_real_dist(robot_initial_positions.get(rid), best_plant_pos) if best_plant_pos else 0
            return (p * 100) + c - d1

        self.primary_id = max(probs, key=robot_selection_score)
        
        # will be used later in planning
        self.action_stack = []
        self.best_Astar_subpath = []
        self.last_state = None
        self.last_action = None
        self.game_start_state = game.get_current_state()
        self.plan_start_state = game.get_current_state()

        # for extra loads and pours
        self.extra_load_count = 0
        self.extra_pour_count = 0

    def _get_robot_data(self, state, rid=None):
        target_id = rid if rid is not None else self.primary_id
        for r_id, pos, load in state[0]:
            if r_id == target_id: return pos, load
        return None, 0

    def _is_pos_occupied(self, state, target_pos):
        for _, pos, _ in state[0]:
            if pos == target_pos: return True
        return False

    def _select_targets(self, state, steps_remaining):
        p_pos, p_load = self._get_robot_data(state)
        plants_list = state[1]
        taps_dict = dict(state[2])
        
        # in smaller world case, we find if its most valuable to water all plants or onl 1
        if self.is_small_world:
            average_reward = sum(self.mean_rewards.values()) / len(self.mean_rewards)
            max_plant = ((0,0) , 0)
            best_plant_flag = False

            for p in plants_list:
                plant_pos = p[0]
                plant_reward = self.mean_rewards[plant_pos]
                if plant_reward > average_reward * 1.5:
                    if plant_reward > max_plant[1]:
                        best_plant_flag = True
                        max_plant = p

            if (best_plant_flag):
                return { max_plant[0] : max_plant[1] }
            else:    
                return {pos: need for pos, need in plants_list if need > 0}

        # in a larger world senario we use  R^2 / (D1 + D2)
        # where R is the mean plant reward, D1 is the distance between the chosen robot
        # and the plant and D2 is the distance between the plant and its closest tap
        best_score = -1
        best_plant_pos = None
        plant_rewards = self.problem_desc.get("plants_reward", {})

        # calculating distances with walls included
        def get_real_dist(start, end):
            if start == end: return 0
            queue = [(start, 0)]
            visited = {start}
            while queue:
                (r, c), dist = queue.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) == end: return dist + 1
                    if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                        (nr, nc) not in self.walls and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
            return 10000  # no path

        for pos, need in plants_list:
            if need > 0:
                # R calculation
                rewards = plant_rewards.get(pos, [1])
                reward_val = sum(rewards) / len(rewards)
                
                # D1 calculation
                d1 = get_real_dist(p_pos, pos)
                if d1 >= 10000: continue # Ignore unreachable plants
                if d1 > steps_remaining + need: continue
                
                # D2 calculation
                d2 = min([get_real_dist(pos, tap) for tap in taps_dict.keys()] + [999])
                
                # final score
                score = (reward_val ** 2) / (d1 + d2 + 1)
                
                if score > best_score:
                    best_score = score
                    best_plant_pos = pos
        
        # calculating extra loads and pours based on the robots fail probability
        if best_plant_pos:
            self.extra_load_count = (1 - self.MAX_PROB) * self.problem_desc["Plants"][best_plant_pos]
            self.extra_pour_count = self.extra_load_count
            return {best_plant_pos: dict(plants_list)[best_plant_pos]}
        
        return {}

    def _solve_problem(self, state, targets_dict, algorithm):
        p_pos, p_load = self._get_robot_data(state)
        sub_prob_dict = {
            "Size": (self.rows, self.cols), 
            "Walls": list(self.walls),
            "Taps": dict(state[2]), 
            "Plants": targets_dict,
            "Robots": {self.primary_id: (p_pos[0], p_pos[1], p_load, self.capacities[self.primary_id])}
        }
        
        # the risk is the robot's probability to fail
        risk = 1.0 - self.MAX_PROB
        extra_ops = 0
        if risk > 0 and targets_dict:
            # 1 extra operation for every 20% risk, minimum 1 if risk exists
            extra_ops = int(risk * 5) + 1 

        try:
            # creating the problem and using ex1 code
            p = WateringProblem(sub_prob_dict)
            result = greedy_best_first_graph_search(p, p.h_astar) if algorithm == 'gbfs' else astar_search(p, p.h_astar)
            node = result[0] if isinstance(result, tuple) else result
            
            if node:
                actions = node.solution() if hasattr(node, 'solution') else [n.action for n in node.path() if n.action]
                clean = ["".join(filter(str.isalpha, str(a).split('{')[0].split('(')[0])).upper() for a in actions]
                
                # checking if all actions are legal
                if clean:
                    act = clean[0]
                    r, c = p_pos
                    is_start_first = False
                    if act in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        tr = (r-1, c) if act == "UP" else (r+1, c) if act == "DOWN" else (r, c-1) if act == "LEFT" else (r, c+1)
                        if 0 <= tr[0] < self.rows and 0 <= tr[1] < self.cols and tr not in self.walls:
                            is_start_first = True
                    elif (act == "LOAD" and p_pos in dict(state[2])) or (act == "POUR" and p_pos in targets_dict):
                        is_start_first = True
                    if not is_start_first: clean = clean[::-1]
                
                # injecting the extra pours and loads, in reality we inject too many but this will be 
                # ignored thorugh the choose_next_action logic
                final_actions = []
                for act in clean:
                    final_actions.append(act)
                    if act == "LOAD":
                        # adding the extra loads
                        for _ in range(extra_ops):
                            final_actions.append("LOAD")
                    elif act == "POUR":
                        # adding the extra pours
                        for _ in range(extra_ops):
                            final_actions.append("POUR")

                return [f"{a}({self.primary_id})" for a in final_actions]
        except: pass
        return []

    def _calculate_optimal_path(self, state, steps_remaining, algorithm):
        # target selection
        targets = self._select_targets(state, steps_remaining)
        self.target_plants = targets.copy()
        if not targets: return []
        # running ex1 with only the target plants
        return self._solve_problem(state, targets, algorithm)

    def recognize_and_fix_fail(self, last_state, last_action, current_state):
        if not last_action or last_action == "RESET":
            return None
        
        act_name = last_action.split('(')[0].upper()
        rid = self.primary_id
        p_prev, l_prev = self._get_robot_data(last_state)
        p_curr, l_curr = self._get_robot_data(current_state)
        
        # movement fail handeling
        if act_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
            r, c = p_prev
            target_pos = p_prev
            if act_name == "UP": target_pos = (r-1, c)
            elif act_name == "DOWN": target_pos = (r+1, c)
            elif act_name == "LEFT": target_pos = (r, c-1)
            elif act_name == "RIGHT": target_pos = (r, c+1)

            if p_curr == target_pos: return None # move succeeded
            if p_curr == p_prev: return last_action # block case

            # unwanted move "fix"
            cr, cc = p_curr
            pr, pc = p_prev
            fix_move = None
            target_sq = None

            if cr < pr: fix_move, target_sq = f"DOWN({rid})", (cr+1, cc)
            elif cr > pr: fix_move, target_sq = f"UP({rid})", (cr-1, cc)
            elif cc < pc: fix_move, target_sq = f"RIGHT({rid})", (cr, cc+1)
            elif cc > pc: fix_move, target_sq = f"LEFT({rid})", (cr, cc-1)

            if fix_move and target_sq:
                if target_sq in self.walls:
                    return "CLEAR_STACK"
                return fix_move

        # when fialing load, we just load again
        elif act_name == "LOAD":
            if l_curr <= l_prev:
                if p_curr not in dict(current_state[2]): return "CLEAR_STACK"
                return last_action
        
        # retry to pour
        elif act_name == "POUR":
            if l_curr >= l_prev:
                plants_dict = dict(current_state[1])
                if l_curr == 0 or p_curr not in plants_dict: return "CLEAR_STACK"
                return last_action

        return None

    def recognize_and_fix_blocking_robot(self, state, next_action):
        p_pos, _ = self._get_robot_data(state)
        
        def get_target(start_pos, act_str):
            name = act_str.split('(')[0].upper()
            r, c = start_pos
            if name == "UP": return (r-1, c)
            if name == "DOWN": return (r+1, c)
            if name == "LEFT": return (r, c-1)
            if name == "RIGHT": return (r, c+1)
            return start_pos

        immediate_target = get_target(p_pos, next_action)
        if immediate_target == p_pos:
            return None

        # we need the future move in order that the blocking robot will continue to block us afterwards
        future_target = None
        if len(self.action_stack) > 1:
            future_target = get_target(immediate_target, self.action_stack[1])

        # find the blocker
        for rid, pos, _ in state[0]:
            if pos == immediate_target and rid != self.primary_id:
                # checking valid moves for the blocker
                for dr, dc, m in [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]:
                    nr, nc = pos[0] + dr, pos[1] + dc
                    
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
                        if not self._is_pos_occupied(state, (nr, nc)):
                            # forbidden moves are where the primary robot is currently, its future move and staying in place
                            forbidden = {p_pos, immediate_target, future_target}
                            if (nr, nc) not in forbidden:
                                return f"{m}({rid})"
                
                # in case we dont have a "perfect" action still try to move to the future path, maybe some tile 
                # will open up later
                for dr, dc, m in [(-1, 0, "UP"), (1, 0, "DOWN"), (0, -1, "LEFT"), (0, 1, "RIGHT")]:
                    nr, nc = pos[0] + dr, pos[1] + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
                        if not self._is_pos_occupied(state, (nr, nc)) and (nr, nc) != p_pos and (nr, nc) != immediate_target:
                            return f"{m}({rid})"

                # if we are "stuck", just reset
                return "RESET"
        return None

    def _calculate_plan_value(self, plan, start_state) :
        if not plan or not start_state: return 0.0

        start_pos, start_load = self._get_robot_data(start_state)
        plants = dict(start_state[1])
        total_plant_needed = len([p for p, n in plants.items() if n > 0])
        plants_watered = 0

        expected_score = 0.0
        move_cost = -3.0 / self.MAX_PROB
        action_cost = -1.0 / self.MAX_PROB
        current_pos = start_pos
        current_load = start_load

        for action_str in plan:
            act_name = action_str.split('(')[0].upper()
            expected_score += action_cost

            if act_name == "UP":
                current_pos = (current_pos[0]-1, current_pos[1])
                expected_score += move_cost
            elif act_name == "DOWN":
                current_pos = (current_pos[0]+1, current_pos[1])
                expected_score += move_cost
            elif act_name == "LEFT":
                current_pos = (current_pos[0], current_pos[1]-1)
                expected_score += move_cost
            elif act_name == "RIGHT":
                current_pos = (current_pos[0], current_pos[1]+1)
                expected_score += move_cost
            elif act_name == "LOAD": current_load = self.capacities[self.primary_id]
            elif act_name == "POUR":
                plant_need = plants.get(current_pos, 0)
                if plant_need > 0 and current_load > 0:
                    # giving reward
                    mean_r = self.mean_rewards.get(current_pos, 0)
                    expected_score += mean_r
                    
                    # decrement need and load for simulation
                    plants[current_pos] = 0 # 
                    current_load -= 1
                    plants_watered += 1
            
        if plants_watered >= total_plant_needed and total_plant_needed > 0:
            expected_score += self.goal_reward

        return expected_score
    

    def choose_next_action(self, state):
        cur_need = state[3]
        steps_remaining = self.game.get_max_steps() - self.game.get_current_steps()
        reset_flag = False

        # applying reset logic
        if (self.last_state and cur_need > self.last_state[3]) or self.last_action == "RESET":
            self.action_stack = []
            self.last_action = None
            self.last_state = None
            self.plan_start_state = self.game_start_state
            reset_flag = True

        # when action_stack is empty, we want to fill it:
        if not self.action_stack:
            # applying extension logic using stored start state
            if not self.is_small_world and not self.extension_check and self.best_Astar_subpath and self.plan_start_state:
                old_reward = self._calculate_plan_value(self.best_Astar_subpath, self.plan_start_state)
                
                # calculate extension
                new_targets = self._select_targets(state, steps_remaining)
                new_path = self._solve_problem(state, new_targets, 'astar')
                
                # reward for whole previous + extension plan
                combined_plan = self.best_Astar_subpath + new_path
                combined_reward = self._calculate_plan_value(combined_plan, self.plan_start_state)
                
                # compare the rewards
                if combined_reward > (old_reward * 2):
                    self.action_stack = new_path
                    # update the best plan
                    self.best_Astar_subpath.extend(new_path)
                else:
                    self.action_stack = self.best_Astar_subpath.copy()
                    self.extension_check = True

            # if we dont have a cached plan, we need to generate it
            if not self.best_Astar_subpath:
                self.action_stack = self._calculate_optimal_path(state, steps_remaining, 'astar') 
                self.best_Astar_subpath = self.action_stack.copy()
                if self.action_stack:
                    self.plan_start_state = copy.deepcopy(state)

            elif reset_flag:
                # when a reset happend, we need to check if we have enough moves to use our plan
                if len(self.best_Astar_subpath) <= steps_remaining * self.MAX_PROB:
                    self.action_stack = self.best_Astar_subpath.copy()
                    self.plan_start_state = copy.deepcopy(state)
                # if we dont have enough moves, we generate a greedy plan for the remaining moves
                else:
                    self.action_stack = self._calculate_optimal_path(state, steps_remaining, 'gbfs') 
                    self.plan_start_state = copy.deepcopy(state)
            
            if not self.action_stack:
                self.last_action = "RESET"
                return "RESET"

        # action fail handeling
        if not reset_flag and self.last_action and self.last_state:
            fix_action = self.recognize_and_fix_fail(self.last_state, self.last_action, state)
            if fix_action == "CLEAR_STACK":
                self.action_stack = []
                self.last_action = "RESET"
                self.plan_start_state = None
                return "RESET" 
            if fix_action:
                return fix_action 

        # we check the next action, and if we want to really perform it
        next_action_str = self.action_stack[0]
        act_name = next_action_str.split('(')[0].upper()
        p_pos, p_load = self._get_robot_data(state)
        plants_dict = dict(state[1])
        
        # we want a couple of "fixed" for the load (can happen because of the extra loads)
        if act_name == "LOAD":
            # if the robot is at max capacity we dont want to load
            if p_load >= self.capacities[self.primary_id]:
                self.action_stack.pop(0)
                return self.choose_next_action(state)
            
            # skip load if we allready loaded enough
            if plants_dict:
                # we want to account for all target plant's needs, and add the chance of fail into account
                max_factor = max(self.target_plants.values()) if self.target_plants else 1
                max_required = max_factor * (1 - self.MAX_PROB) + max_factor
                if p_load >= max_required:
                    self.action_stack.pop(0)
                    return self.choose_next_action(state)
            
            # invalid load case
            if p_pos not in dict(state[2]):
                self.action_stack = []
                self.last_action = "RESET"
                self.plan_start_state = None
                return "RESET"

        elif act_name == "POUR":
            plant_missing = p_pos not in plants_dict
            plant_need = plants_dict.get(p_pos, 0)
            
            # if we have no water left of the plant is fully watered, we should skip
            is_empty = (p_load == 0)
            is_satisfied = plant_missing or (plant_need == 0)
            
            if is_empty or is_satisfied:
                self.action_stack.pop(0)
                if not self.action_stack:
                    self.last_action = "RESET"
                    self.plan_start_state = None
                    return "RESET"
                return self.choose_next_action(state)
        
        # safety check for off-board movement
        elif act_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
            r, c = p_pos
            nr, nc = r, c
            if act_name == "UP": nr -= 1
            elif act_name == "DOWN": nr += 1
            elif act_name == "LEFT": nc -= 1
            elif act_name == "RIGHT": nc += 1
            
            if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (nr, nc) in self.walls:
                self.action_stack = []
                self.last_action = "RESET"
                self.plan_start_state = None
                return "RESET"

        # blocking robot hendeling
        unblock_action = self.recognize_and_fix_blocking_robot(state, next_action_str)
        if unblock_action:
            if unblock_action == "RESET":
                self.action_stack = []
                self.last_action = "RESET"
                self.plan_start_state = None
                return "RESET"
            return unblock_action

        # excuting action and updating fields
        action = self.action_stack.pop(0)
        self.last_state = state
        self.last_action = action
        return action
    
class Problem:
    """The abstract class for a formal problem.  You should subclass this and
    implement the method successor, and possibly __init__, goal_test, and
    path_cost. Then you will create instances of your subclass and solve them
    with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """Given a state, return a sequence of (action, state) pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework."""

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal, as specified in the constructor. Implement this
        method if checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""

# ______________________________________________________________________________

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        update(self, state=state, parent=parent, action=action,
               path_cost=path_cost, depth=0)
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def path(self):
        "Create a list of nodes from the root to this node."
        x, result = self, [self]
        while x.parent:
            result.append(x.parent)
            x = x.parent
        return result

    def expand(self, problem):
        "Return a list of nodes reachable from this node. [Fig. 3.8]"
        return [Node(next, self, act,
                     problem.path_cost(self.path_cost, self.state, act, next))
                for (act, next) in problem.successor(self.state)]

    def __eq__(self, other):
        return (self.f == other.f)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return (self.f < other.f)

    def __gt__(self, other):
        return (self.f > other.f)

    def __le__(self, other):
        return (self < other) or (self == other)

    def __ge__(self, other):
        return (self > other) or (self == other)


def _manhattan_dist(pos1, pos2):
    # Calculating the manhetten distances
    r1, c1 = pos1
    r2, c2 = pos2
    return abs(r1 - r2) + abs(c1 - c2)

class WateringProblem(Problem):
    def __init__(self, initial_problem_description): 
        # The init function initiallizes, static data stored in self and dynamic data in initial state
        self.max_r, self.max_c = initial_problem_description["Size"]
        self.walls = {(r, c) for (r, c) in initial_problem_description["Walls"]}
        
        # Tap plant and robot data
        taps = [(t[0], t[1]) for t in initial_problem_description["Taps"].items()]
        self.taps_positions = [t[0] for t in taps]
        initial_taps = [t[1] for t in taps]

        plants = [(p[0], p[1]) for p in initial_problem_description["Plants"].items()]
        self.plants_positions = [t[0] for t in plants]
        initial_plants = [p[1] for p in plants]

        robots = [(r[0], r[1]) for r in initial_problem_description["Robots"].items()]
        self.robots_ids = [r[0] for r in robots]
        initial_robots = [r[1] for r in robots]
        
        self.tap_index_map = {pos:i for i,pos in enumerate(self.taps_positions)}
        self.plant_index_map = {pos:i for i,pos in enumerate(self.plants_positions)}
        
        # String caching
        self.cached_loads = [f"LOAD{{{rid}}}" for rid in self.robots_ids]
        self.cached_pours = [f"POUR{{{rid}}}" for rid in self.robots_ids]
        self.cached_moves = []
        for rid in self.robots_ids:
            self.cached_moves.append({
                "UP": f"UP{{{rid}}}", "DOWN": f"DOWN{{{rid}}}",
                "LEFT": f"LEFT{{{rid}}}", "RIGHT": f"RIGHT{{{rid}}}"
            })

        self.moves_list = (
            ("UP", -1, 0), ("DOWN", 1, 0),
            ("LEFT", 0, -1), ("RIGHT", 0, 1)
        )
        
        # Distance chaching
        self._tap_plant_distances = {}
        for t_pos in self.taps_positions:
            for p_pos in self.plants_positions:
                self._tap_plant_distances[(t_pos, p_pos)] = _manhattan_dist(t_pos, p_pos)
        
        # Optemization for single robot
        self.num_robots = len(self.robots_ids)
        if self.num_robots == 1:
            self.single_robot_capacity = initial_robots[0][3]
            self.single_robot_index = 0
        
        initial_state = (tuple(initial_plants), tuple(initial_taps), tuple(initial_robots))
        Problem.__init__(self, initial_state)
        
        # A* cache (for the heuristic)
        self._h_astar_cache = {}
        
        # Max capacity
        self.max_capacity = max((cap for _, _, _, cap in initial_robots), default=1)

    def successor(self, state):
        plants, taps, robots = state
        new_successors = []
        
        is_single_robot = self.num_robots == 1
        
        # When there are multiple robot store occupied robot positions
        if not is_single_robot:
            occupied_positions = {(r, c) for r, c, _, _ in robots}

        for i, (row, col, carry, capacity) in enumerate(robots):
            robot_pos = (row, col)
            action_performed = False

            # LOAD action
            if robot_pos in self.tap_index_map and carry < capacity and carry < sum(p for p in plants):
                tap_index = self.tap_index_map[robot_pos]
                cur_tap_amount = taps[tap_index]
                
                if cur_tap_amount > 0:
                    new_robot = (row, col, carry + 1, capacity)
                    new_robots = robots[:i] + (new_robot,) + robots[i+1:]
                    new_taps = taps[:tap_index] + (cur_tap_amount - 1,) + taps[tap_index+1:]
                    new_state = (plants, new_taps, new_robots)
                    action_str = self.cached_loads[i]
                    if new_state not in self._h_astar_cache:
                        new_successors.append((action_str, new_state))
                        if is_single_robot: # single tobot case - if load is available this is what we will do
                            return new_successors 
                    action_performed = True

            # POUR action
            if robot_pos in self.plant_index_map and carry > 0:
                plant_index = self.plant_index_map[robot_pos]
                cur_plant_need = plants[plant_index]
                
                if cur_plant_need > 0:
                    new_robot = (row, col, carry - 1, capacity)
                    new_robots = robots[:i] + (new_robot,) + robots[i+1:]
                    new_plants = plants[:plant_index] + (cur_plant_need - 1,) + plants[plant_index+1:]
                    new_state = (new_plants, taps, new_robots)
                    action_str = self.cached_pours[i]
                    if new_state not in self._h_astar_cache:
                        new_successors.append((action_str, new_state)) 
                        if is_single_robot:
                            return new_successors # If pour is available, this is the action
                    action_performed = True

            # Move actions
            if not is_single_robot or not action_performed:
                for move_name, dr, dc in self.moves_list:
                    new_r, new_c = row + dr, col + dc
                        
                    # Boundaries check
                    if (0 <= new_r < self.max_r and 0 <= new_c < self.max_c and
                        (new_r, new_c) not in self.walls):
                        
                        # Occupied position check
                        if is_single_robot or (new_r, new_c) not in occupied_positions:
                            new_robot = (new_r, new_c, carry, capacity)
                            new_robots = robots[:i] + (new_robot,) + robots[i+1:]
                            new_state = (plants, taps, new_robots)
                            action_str = self.cached_moves[i][move_name]
                            if new_state not in self._h_astar_cache:
                                new_successors.append((action_str, new_state))

        return new_successors

    def goal_test(self, state):
        plants_tuple, _, _ = state
        return all(need == 0 for need in plants_tuple)

    def h_astar(self, node):
        state = node.state
        if state in self._h_astar_cache:
            return self._h_astar_cache[state]

        plants_needs, taps_amounts, robots_state = state
        
        # Check if the goal state is reached
        total_remaining_need = sum(plants_needs)
        if total_remaining_need == 0:
            self._h_astar_cache[state] = 0
            return 0

        # Counting load + pour actions needed
        min_pour_actions = total_remaining_need
        total_carried_water = sum(carry for _, _, carry, _ in robots_state)
        min_load_actions = max(0, total_remaining_need - total_carried_water)
        action_lower_bound = min_pour_actions + min_load_actions

        # Movement count
        movement_lower_bound = 0 
        active_taps_positions = tuple(self.taps_positions[i] for i, amount in enumerate(taps_amounts) if amount > 0)
        unwatered_plants_positions = [self.plants_positions[i] for i, need in enumerate(plants_needs) if need > 0]
        
        is_single_robot = self.num_robots == 1
        
        if is_single_robot:
            # Custom heuristic for single robot
            r_row, r_col, carry, _ = robots_state[0]
            robot_pos = (r_row, r_col)
            
            # Adding the maximum distance between 2 unwattered plants
            max_p_to_p_dist = 0
            if len(unwatered_plants_positions) >= 2:
                for i in range(len(unwatered_plants_positions)):
                    for j in range(i + 1, len(unwatered_plants_positions)):
                        p1_pos = unwatered_plants_positions[i]
                        p2_pos = unwatered_plants_positions[j]
                        max_p_to_p_dist = max(max_p_to_p_dist, _manhattan_dist(p1_pos, p2_pos))
            
                movement_lower_bound += max_p_to_p_dist
            
        else: # Multi-Robot Case, using max/min logic - the maximum distance between all the minimum distances robots need to travel
            for p_idx, need in enumerate(plants_needs):
                if need > 0:
                    plant_pos = self.plants_positions[p_idx]
                    min_cost_to_service_plant = infinity
                    
                    for r_row, r_col, carry, _ in robots_state:
                        robot_pos = (r_row, r_col)
                        cost = infinity
                        
                        if carry > 0:
                            cost = _manhattan_dist(robot_pos, plant_pos)
                        elif active_taps_positions:
                            # R -> T -> P (Using cached T->P distance)
                            cost = min(
                                _manhattan_dist(robot_pos, tap_pos) + self._tap_plant_distances[(tap_pos, plant_pos)]
                                for tap_pos in active_taps_positions
                            )
                        min_cost_to_service_plant = min(min_cost_to_service_plant, cost)
                    
                    movement_lower_bound = max(movement_lower_bound, min_cost_to_service_plant) 

        # Summing the bounds
        h_value = action_lower_bound + movement_lower_bound
        self._h_astar_cache[state] = h_value
        return h_value

    def h_gbfs(self, node):
        state = node.state
        plants_needs, taps_amounts, robots_state = state
        
        total_remaining_need = sum(plants_needs)
        if total_remaining_need == 0:
            return 0

        # Counting load + pour actions
        total_pours_needed = total_remaining_need
        total_carried_water = sum(carry for _, _, carry, _ in robots_state)
        total_loads_needed = max(0, total_remaining_need - total_carried_water)
        
        # Minimum next action distance 
        min_next_action_distance = infinity
        active_taps_positions = tuple(self.taps_positions[i] for i, amount in enumerate(taps_amounts) if amount > 0)
        unwatered_plants_positions = tuple(self.plants_positions[i] for i, need in enumerate(plants_needs) if need > 0)
        
        is_single_robot = self.num_robots == 1

        if is_single_robot: # Single robot case
            r_row, r_col, carry, _ = robots_state[0]
            robot_pos = (r_row, r_col)
            
            # If we have water, we need to travel to a plant.
            if carry > 0 and unwatered_plants_positions:
                min_next_action_distance = min(
                    _manhattan_dist(robot_pos, p_pos) 
                    for p_pos in unwatered_plants_positions
                )
            elif active_taps_positions and unwatered_plants_positions:
                min_next_action_distance = min(
                    _manhattan_dist(robot_pos, tap_pos) + self._tap_plant_distances[(tap_pos, p_pos)]
                    for tap_pos in active_taps_positions
                    for p_pos in unwatered_plants_positions
                )
        
        else: # Multi Robot Case 
            # calcutaing the minimal active tap to plant
            min_active_tap_to_plant_dist_for_dry_robot = infinity
            if active_taps_positions and unwatered_plants_positions:
                min_active_tap_to_plant_dist_for_dry_robot = min(
                    self._tap_plant_distances[(tap_pos, p_pos)]
                    for tap_pos in active_taps_positions
                    for p_pos in unwatered_plants_positions
                )
            
            for r_row, r_col, carry, _ in robots_state:
                robot_pos = (r_row, r_col)
                current_robot_best_path = infinity

                if carry > 0 and unwatered_plants_positions:
                    current_robot_best_path = min(
                        _manhattan_dist(robot_pos, p_pos) 
                        for p_pos in unwatered_plants_positions
                    )
                elif active_taps_positions:
                    min_robot_to_tap_dist = min(
                        _manhattan_dist(robot_pos, tap_pos) for tap_pos in active_taps_positions
                    )
                    
                    if min_active_tap_to_plant_dist_for_dry_robot != infinity:
                        current_robot_best_path = min_robot_to_tap_dist + min_active_tap_to_plant_dist_for_dry_robot
                
                min_next_action_distance = min(min_next_action_distance, current_robot_best_path)

        # Weighted calculation
        load_component = total_loads_needed * 10000
        pour_component = total_pours_needed * 100
        distance_component = min_next_action_distance if min_next_action_distance != infinity else 0

        h_value = load_component + pour_component + distance_component
        return h_value

def create_watering_problem(game):
    return WateringProblem(game)

# ______________________________________________________________________________
## Uninformed Search algorithms

def tree_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    Don't worry about repeated paths to a state. [Fig. 3.8]"""
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None


def breadth_first_tree_search(problem):
    "Search the shallowest nodes in the search tree first. [p 74]"
    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):
    "Search the deepest nodes in the search tree first. [p 74]"
    return tree_search(problem, Stack())


def graph_search(problem, fringe):
    """Search through the successors of a problem to find a goal.
    The argument fringe should be an empty queue.
    If two paths reach a state, only use the best one. [Fig. 3.18]"""
    closed = {}
    expanded = 0
   
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node, expanded
        if node.state not in closed:
            closed[node.state] = True
            fringe.extend(node.expand(problem))
            expanded += 1
    return None


def breadth_first_graph_search(problem):
    "Search the shallowest nodes in the search tree first. [p 74]"
    return graph_search(problem, FIFOQueue())


def depth_first_graph_search(problem):
    "Search the deepest nodes in the search tree first. [p 74]"
    return graph_search(problem, Stack())


def depth_limited_search(problem, limit=50):
    "[Fig. 3.12]"

    def recursive_dls(node, problem, limit):
        cutoff_occurred = False
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            for successor in node.expand(problem):
                result = recursive_dls(successor, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result != None:
                    return result
        if cutoff_occurred:
            return 'cutoff'
        else:
            return None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    "[Fig. 3.13]"
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result


# ______________________________________________________________________________
# Informed (Heuristic) Search

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have depth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    return graph_search(problem, PriorityQueue(min, f))


greedy_best_first_graph_search = best_first_graph_search


# Greedy best-first search is accomplished by specifying f(n) = h(n).

def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search.
    Uses the pathmax trick: f(n) = max(f(n), g(n)+h(n))."""
    h = h or problem.h

    def f(n):
        return max(getattr(n, 'f', -infinity), n.path_cost + h(n))

    return best_first_graph_search(problem, f)


# ______________________________________________________________________________
## Other search algorithms

def recursive_best_first_search(problem):
    "[Fig. 4.5]"

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node
        successors = expand(node, problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + s.h, node.f)
        while True:
            successors.sort(lambda x, y: x.f - y.f)  # Order by lowest f value
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            alternative = successors[1]
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result

    return RBFS(Node(problem.initial), infinity)


def hill_climbing(problem):
    """From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better. [Fig. 4.11]"""
    current = Node(problem.initial)
    while True:
        neighbor = argmax(expand(node, problem), Node.value)
        if neighbor.value() <= current.value():
            return current.state
        current = neighbor


def exp_schedule(k=20, lam=0.005, limit=100):
    "One possible schedule function for simulated annealing"
    return lambda t: if_(t < limit, k * math.exp(-lam * t), 0)


def simulated_annealing(problem, schedule=exp_schedule()):
    "[Fig. 4.5]"
    current = Node(problem.initial)
    for t in xrange(sys.maxint):
        T = schedule(t)
        if T == 0:
            return current
        next = random.choice(expand(node.problem))
        delta_e = next.path_cost - current.path_cost
        if delta_e > 0 or probability(math.exp(delta_e / T)):
            current = next


def online_dfs_agent(a):
    "[Fig. 4.12]"
    pass  #### more


def lrta_star_agent(a):
    "[Fig. 4.12]"
    pass  #### more

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)


# ______________________________________________________________________________
# Compatibility with Python 2.2 and 2.3

# The AIMA code is designed to run in Python 2.2 and up (at some point,
# support for 2.2 may go away; 2.2 was released in 2001, and so is over
# 3 years old). The first part of this file brings you up to 2.4
# compatibility if you are running in Python 2.2 or 2.3:

# try: bool, True, False ## Introduced in 2.3
# except NameError:
#     class bool(int):
#         "Simple implementation of Booleans, as in PEP 285"
#         def __init__(self, val): self.val = val
#         def __int__(self): return self.val
#         def __repr__(self): return ('False', 'True')[self.val]
#
#     True, False = bool(1), bool(0)
#
# try: sum ## Introduced in 2.3
# except NameError:
#     def sum(seq, start=0):
#         """Sum the elements of seq.
#         >>> sum([1, 2, 3])
#         6
#         """
#         return reduce(operator.add, seq, start)

try:
    enumerate  ## Introduced in 2.3
except NameError:
    def enumerate(collection):
        """Return an iterator that enumerates pairs of (i, c[i]). PEP 279.
        >>> list(enumerate('abc'))
        [(0, 'a'), (1, 'b'), (2, 'c')]
        """
        ## Copied from PEP 279
        i = 0
        it = iter(collection)
        while 1:
            yield (i, it.next())
            i += 1

try:
    reversed  ## Introduced in 2.4
except NameError:
    def reversed(seq):
        """Iterate over x in reverse order.
        >>> list(reversed([1,2,3]))
        [3, 2, 1]
        """
        if hasattr(seq, 'keys'):
            raise ValueError("mappings do not support reverse iteration")
        i = len(seq)
        while i > 0:
            i -= 1
            yield seq[i]

try:
    sorted  ## Introduced in 2.4
except NameError:
    def sorted(seq, cmp=None, key=None, reverse=False):
        """Copy seq and sort and return it.
        >>> sorted([3, 1, 2])
        [1, 2, 3]
        """
        seq2 = copy.copy(seq)
        if key:
            if cmp == None:
                cmp = __builtins__.cmp
            seq2.sort(lambda x, y: cmp(key(x), key(y)))
        else:
            if cmp == None:
                seq2.sort()
            else:
                seq2.sort(cmp)
        if reverse:
            seq2.reverse()
        return seq2

try:
    set, frozenset  ## set builtin introduced in 2.4
except NameError:
    try:
        import sets  ## sets module introduced in 2.3

        set, frozenset = sets.Set, sets.ImmutableSet
    except (NameError, ImportError):
        class BaseSet:
            "set type (see http://docs.python.org/lib/types-set.html)"

            def __init__(self, elements=[]):
                self.dict = {}
                for e in elements:
                    self.dict[e] = 1

            def __len__(self):
                return len(self.dict)

            def __iter__(self):
                for e in self.dict:
                    yield e

            def __contains__(self, element):
                return element in self.dict

            def issubset(self, other):
                for e in self.dict.keys():
                    if e not in other:
                        return False
                return True

            def issuperset(self, other):
                for e in other:
                    if e not in self:
                        return False
                return True

            def union(self, other):
                return type(self)(list(self) + list(other))

            def intersection(self, other):
                return type(self)([e for e in self.dict if e in other])

            def difference(self, other):
                return type(self)([e for e in self.dict if e not in other])

            def symmetric_difference(self, other):
                return type(self)([e for e in self.dict if e not in other] +
                                  [e for e in other if e not in self.dict])

            def copy(self):
                return type(self)(self.dict)

            def __repr__(self):
                elements = ", ".join(map(str, self.dict))
                return "%s([%s])" % (type(self).__name__, elements)

            __le__ = issubset
            __ge__ = issuperset
            __or__ = union
            __and__ = intersection
            __sub__ = difference
            __xor__ = symmetric_difference


        class frozenset(BaseSet):
            "A frozenset is a BaseSet that has a hash value and is immutable."

            def __init__(self, elements=[]):
                BaseSet.__init__(elements)
                self.hash = 0
                for e in self:
                    self.hash |= hash(e)

            def __hash__(self):
                return self.hash


        class set(BaseSet):
            "A set is a BaseSet that does not have a hash, but is mutable."

            def update(self, other):
                for e in other:
                    self.add(e)
                return self

            def intersection_update(self, other):
                for e in self.dict.keys():
                    if e not in other:
                        self.remove(e)
                return self

            def difference_update(self, other):
                for e in self.dict.keys():
                    if e in other:
                        self.remove(e)
                return self

            def symmetric_difference_update(self, other):
                to_remove1 = [e for e in self.dict if e in other]
                to_remove2 = [e for e in other if e in self.dict]
                self.difference_update(to_remove1)
                self.difference_update(to_remove2)
                return self

            def add(self, element):
                self.dict[element] = 1

            def remove(self, element):
                del self.dict[element]

            def discard(self, element):
                if element in self.dict:
                    del self.dict[element]

            def pop(self):
                key, val = self.dict.popitem()
                return key

            def clear(self):
                self.dict.clear()

            __ior__ = update
            __iand__ = intersection_update
            __isub__ = difference_update
            __ixor__ = symmetric_difference_update

# ______________________________________________________________________________
# Simple Data Structures: infinity, Dict, Struct

infinity = 1.0e400


def Dict(**entries):
    """Create a dict out of the argument=value arguments. 
    >>> Dict(a=1, b=2, c=3)
    {'a': 1, 'c': 3, 'b': 2}
    """
    return entries


class DefaultDict(dict):
    """Dictionary with a default value for unknown keys."""

    def __init__(self, default):
        self.default = default

    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, copy.deepcopy(self.default))

    def __copy__(self):
        copy = DefaultDict(self.default)
        copy.update(self)
        return copy


class Struct:
    """Create an instance with argument=value slots.
    This is for making a lightweight object whose class doesn't matter."""

    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __cmp__(self, other):
        if isinstance(other, Struct):
            return cmp(self.__dict__, other.__dict__)
        else:
            return cmp(self.__dict__, other)

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k, v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(args)


def update(x, **entries):
    """Update a dict; or an object with slots; according to entries.
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    >>> update(Struct(a=1), a=10, b=20)
    Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries)
    return x


# ______________________________________________________________________________
# Functions on Sequences (mostly inspired by Common Lisp)
# NOTE: Sequence functions (count_if, find_if, every, some) take function
# argument first (like reduce, filter, and map).

def removeall(item, seq):
    """Return a copy of seq (or string) with all occurences of item removed.
    >>> removeall(3, [1, 2, 3, 3, 2, 1, 3])
    [1, 2, 2, 1]
    >>> removeall(4, [1, 2, 3])
    [1, 2, 3]
    """
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]


def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements.
    >>> unique([1, 2, 3, 2, 1])
    [1, 2, 3]
    """
    return list(set(seq))


def product(numbers):
    """Return the product of the numbers.
    >>> product([1,2,3,4])
    24
    """
    return reduce(operator.mul, numbers, 1)


def count_if(predicate, seq):
    """Count the number of elements of seq for which the predicate is true.
    >>> count_if(callable, [42, None, max, min])
    2
    """
    f = lambda count, x: count + (not not predicate(x))
    return reduce(f, seq, 0)


def find_if(predicate, seq):
    """If there is an element of seq that satisfies predicate; return it.
    >>> find_if(callable, [3, min, max])
    <built-in function min>
    >>> find_if(callable, [1, 2, 3])
    """
    for x in seq:
        if predicate(x): return x
    return None


def every(predicate, seq):
    """True if every element of seq satisfies predicate.
    >>> every(callable, [min, max])
    1
    >>> every(callable, [min, 3])
    0
    """
    for x in seq:
        if not predicate(x): return False
    return True


def some(predicate, seq):
    """If some element x of seq satisfies predicate(x), return predicate(x).
    >>> some(callable, [min, 3])
    1
    >>> some(callable, [2, 3])
    0
    """
    for x in seq:
        px = predicate(x)
        if px: return px
    return False


def isin(elt, seq):
    """Like (elt in seq), but compares with is, not ==.
    >>> e = []; isin(e, [1, e, 3])
    True
    >>> isin(e, [1, [], 3])
    False
    """
    for x in seq:
        if elt is x: return True
    return False


# ______________________________________________________________________________
# Functions on sequences of numbers
# NOTE: these take the sequence argument first, like min and max,
# and like standard math notation: \sigma (i = 1..n) fn(i)
# A lot of programing is finding the best value that satisfies some condition;
# so there are three versions of argmin/argmax, depending on what you want to
# do with ties: return the first one, return them all, or pick at random.


def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0];
    best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best


def argmin_list(seq, fn):
    """Return a list of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """
    best_score, best = fn(seq[0]), []
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = [x], x_score
        elif x_score == best_score:
            best.append(x)
    return best


def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]);
    n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score;
            n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                best = x
    return best


def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))


def argmax_list(seq, fn):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    return argmin_list(seq, lambda x: -fn(x))


def argmax_random_tie(seq, fn):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmin_random_tie(seq, lambda x: -fn(x))


# ______________________________________________________________________________
# Statistical and mathematical functions

def histogram(values, mode=0, bin_function=None):
    """Return a list of (value, count) pairs, summarizing the input values.
    Sorted by increasing value, or if mode=1, by decreasing count.
    If bin_function is given, map it over values first."""
    if bin_function: values = map(bin_function, values)
    bins = {}
    for val in values:
        bins[val] = bins.get(val, 0) + 1
    if mode:
        return sorted(bins.items(), key=lambda v: v[1], reverse=True)
    else:
        return sorted(bins.items())


def log2(x):
    """Base 2 logarithm.
    >>> log2(1024)
    10.0
    """
    return math.log10(x) / math.log10(2)


def mode(values):
    """Return the most common value in the list of values.
    >>> mode([1, 2, 3, 2])
    2
    """
    return histogram(values, mode=1)[0][0]


def median(values):
    """Return the middle value, when the values are sorted.
    If there are an odd number of elements, try to average the middle two.
    If they can't be averaged (e.g. they are strings), choose one at random.
    >>> median([10, 100, 11])
    11
    >>> median([1, 2, 3, 4])
    2.5
    """
    n = len(values)
    values = sorted(values)
    if n % 2 == 1:
        return values[n / 2]
    else:
        middle2 = values[(n / 2) - 1:(n / 2) + 1]
        try:
            return mean(middle2)
        except TypeError:
            return random.choice(middle2)


def mean(values):
    """Return the arithmetic average of the values."""
    return sum(values) / float(len(values))


def stddev(values, meanval=None):
    """The standard deviation of a set of values.
    Pass in the mean if you already know it."""
    if meanval == None: meanval = mean(values)
    return math.sqrt(sum([(x - meanval) ** 2 for x in values]) / (len(values) - 1))


def dotproduct(X, Y):
    """Return the sum of the element-wise product of vectors x and y.
    >>> dotproduct([1, 2, 3], [1000, 100, 10])
    1230
    """
    return sum([x * y for x, y in zip(X, Y)])


def vector_add(a, b):
    """Component-wise addition of two vectors.
    >>> vector_add((0, 1), (8, 9))
    (8, 10)
    """
    return tuple(map(operator.add, a, b))


def probability(p):
    "Return true with probability p."
    return p > random.uniform(0.0, 1.0)


def num_or_str(x):
    """The argument is a string; convert to a number if possible, or strip it.
    >>> num_or_str('42')
    42
    >>> num_or_str(' 42x ')
    '42x'
    """
    if isnumber(x): return x
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def normalize(numbers, total=1.0):
    """Multiply each number by a constant such that the sum is 1.0 (or total).
    >>> normalize([1,2,1])
    [0.25, 0.5, 0.25]
    """
    k = total / sum(numbers)
    return [k * n for n in numbers]


## OK, the following are not as widely useful utilities as some of the other
## functions here, but they do show up wherever we have 2D grids: Wumpus and
## Vacuum worlds, TicTacToe and Checkers, and markov decision Processes.

orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def turn_right(orientation):
    return orientations[orientations.index(orientation) - 1]


def turn_left(orientation):
    return orientations[(orientations.index(orientation) + 1) % len(orientations)]


def distance(t1, t2):  # (ax, ay), (bx, by)):
    "The distance between two (x, y) points."
    return math.hypot((t1.ax - t2.bx), (t1.ay - t2.by))


def distance2(t1, t2):  # ((ax, ay), (bx, by)):
    "The square of the distance between two (x, y) points."
    return (t1.ax - t2.bx) ** 2 + (t1.ay - t2.by) ** 2


def clip(vector, lowest, highest):
    """Return vector, except if any element is less than the corresponding
    value of lowest or more than the corresponding value of highest, clip to
    those values.
    >>> clip((-1, 10), (0, 0), (9, 9))
    (0, 9)
    """
    return type(vector)(map(min, map(max, vector, lowest), highest))


# ______________________________________________________________________________
# Misc Functions

def printf(format, *args):
    """Format args with the first argument as format string, and write.
    Return the last arg, or format itself if there are no args."""
    sys.stdout.write(str(format) % args)
    return if_(args, args[-1], format)


def caller(n=1):
    """Return the name of the calling function n levels up in the frame stack.
    >>> caller(0)
    'caller'
    >>> def f(): 
    ...     return caller()
    >>> f()
    'f'
    """
    import inspect
    return inspect.getouterframes(inspect.currentframe())[n][3]


def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


def if_(test, result, alternative):
    """Like C++ and Java's (test ? result : alternative), except
    both result and alternative are always evaluated. However, if
    either evaluates to a function, it is applied to the empty arglist,
    so you can delay execution by putting it in a lambda.
    >>> if_(2 + 2 == 4, 'ok', lambda: expensive_computation())
    'ok'
    """
    if test:
        if callable(result): return result()
        return result
    else:
        if callable(alternative): return alternative()
        return alternative


def name(object):
    "Try to find some reasonable name for the object."
    return (getattr(object, 'name', 0) or getattr(object, '__name__', 0)
            or getattr(getattr(object, '__class__', 0), '__name__', 0)
            or str(object))


def isnumber(x):
    "Is x a number? We say it is if it has a __int__ method."
    return hasattr(x, '__int__')


def issequence(x):
    "Is x a sequence? We say it is if it has a __getitem__ method."
    return hasattr(x, '__getitem__')


def print_table(table, header=None, sep=' ', numfmt='%g'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '%6.2f'.
    (If you want different formats in differnt columns, don't use print_table.)
    sep is the separator between columns."""
    justs = [if_(isnumber(x), 'rjust', 'ljust') for x in table[0]]
    if header:
        table = [header] + table
    table = [[if_(isnumber(x), lambda: numfmt % x, x) for x in row]
             for row in table]
    maxlen = lambda seq: max(map(len, seq))
    sizes = map(maxlen, zip(*[map(str, row) for row in table]))
    for row in table:
        for (j, size, x) in zip(justs, sizes, row):
            print(getattr(str(x), j)(size), sep),
        print()


def AIMAFile(components, mode='r'):
    "Open a file based at the AIMA root directory."
    import utils
    dir = os.path.dirname(utils.__file__)
    return open(apply(os.path.join, [dir] + components), mode)


def DataFile(name, mode='r'):
    "Return a file in the AIMA /data directory."
    return AIMAFile(['..', 'data', name], mode)


# ______________________________________________________________________________
# Queues: Stack, FIFOQueue, PriorityQueue

class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(lt): Queue where items are sorted by lt, (default <).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        abstract

    def extend(self, items):
        for item in items: self.append(item)


def Stack():
    """Return an empty list, suitable as a Last-In-First-Out Queue."""
    return []


class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""

    def __init__(self):
        self.A = [];
        self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A) / 2:
            self.A = self.A[self.start:]
            self.start = 0
        return e


class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x)."""

    def __init__(self, order=min, f=lambda x: x):
        update(self, A=[], order=order, f=f)

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]


## Fig: The idea is we can define things like Fig[3,10] later.
## Alas, it is Fig[3,10] not Fig[3.10], because that would be the same as Fig[3.1]
Fig = {}
