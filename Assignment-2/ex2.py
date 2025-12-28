import re
import collections
import ext_plant 

class Controller:
    def __init__(self, game: ext_plant.Game):
        self.game = game
        self.problem = game.get_problem()
        self.rows, self.cols = self.problem["Size"]
        self.walls = set(game.walls)
        self.capacities = game.get_capacities()
        self.robot_probs = self.problem.get("robot_chosen_action_prob", {})
        
        # Precompute BFS Distances
        self._grid_distances = self._compute_grid_distances()
        
        # Precompute expected plant rewards
        self.expected_rewards = { 
            pos: sum(p for p in rewards) / len(rewards) 
            for pos, rewards in self.problem["plants_reward"].items() 
        }
        
        # Increase Depth for this horizon
        self.MAX_DEPTH = 3 
        self._value_cache = {}

    def _compute_grid_distances(self):
        dist_map = {}
        nodes = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.walls]
        for start in nodes:
            queue = collections.deque([(start, 0)])
            visited = {start}
            while queue:
                curr, d = queue.popleft()
                dist_map[(start, curr)] = d
                r, c = curr
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nxt = (r+dr, c+dc)
                    if 0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols and nxt not in self.walls and nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, d+1))
        return dist_map

    def heuristic(self, state):
        robots_t, plants_t, taps_t, total_need = state
        if total_need == 0: return 1000
        
        score = 0
        # 1. Major penalty for remaining water need (the "Goal" pull)
        score -= total_need * 200 

        for rid, r_pos, r_load in robots_t:
            capacity = self.capacities[rid]
            
            # 2. Reward carrying water (encourages LOAD even if POUR is far away)
            score += r_load * 15 
            
            if r_load < capacity and any(t[1] > 0 for t in taps_t):
                # Needs water: Pull to tap
                dists = [self._grid_distances.get((r_pos, t_pos), 100) for t_pos, t_amt in taps_t if t_amt > 0]
                if dists: score -= min(dists) * 10
            
            if r_load > 0:
                # Has water: Pull to plant
                dists = [self._grid_distances.get((r_pos, p_pos), 100) for p_pos, p_need in plants_t]
                if dists: score -= min(dists) * 10
        
        return score

    def successor(self, state):
        robots_t, plants_t, taps_t, total_need = state
        successors = []
        occupied = {r[1] for r in robots_t}
        
        for i, (rid, r_pos, r_load) in enumerate(robots_t):
            # MOVE
            for m_name, dr, dc in [("UP",-1,0), ("DOWN",1,0), ("LEFT",0,-1), ("RIGHT",0,1)]:
                nr, nc = r_pos[0]+dr, r_pos[1]+dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                    (nr, nc) not in self.walls and (nr, nc) not in occupied):
                    new_robots = list(robots_t)
                    new_robots[i] = (rid, (nr, nc), r_load)
                    successors.append((f"{m_name}({rid})", (tuple(new_robots), plants_t, taps_t, total_need)))
            
            # LOAD
            for t_idx, (t_pos, t_amt) in enumerate(taps_t):
                if r_pos == t_pos and r_load < self.capacities[rid] and t_amt > 0:
                    new_robots = list(robots_t); new_robots[i] = (rid, r_pos, r_load + 1)
                    new_taps = list(taps_t)
                    if t_amt > 1: new_taps[t_idx] = (t_pos, t_amt - 1)
                    else: new_taps.pop(t_idx)
                    successors.append((f"LOAD({rid})", (tuple(new_robots), plants_t, tuple(new_taps), total_need)))
            
            # POUR
            for p_idx, (p_pos, p_need) in enumerate(plants_t):
                if r_pos == p_pos and r_load > 0 and p_need > 0:
                    new_robots = list(robots_t); new_robots[i] = (rid, r_pos, r_load - 1)
                    new_plants = list(plants_t)
                    if p_need > 1: new_plants[p_idx] = (p_pos, p_need - 1)
                    else: new_plants.pop(p_idx)
                    successors.append((f"POUR({rid})", (tuple(new_robots), tuple(new_plants), taps_t, total_need - 1)))
        return successors

    def value(self, state, depth):
        state_key = (state, depth)
        if state_key in self._value_cache: return self._value_cache[state_key]
        if state[3] == 0: return 1000
        if depth == 0: return self.heuristic(state)

        possible = self.successor(state)
        if not possible: return -2000
        
        best_expected_val = -float('inf')
        for action, succ_state in possible:
            rid = int(re.findall(r'\((\d+)\)', action)[0])
            prob = self.robot_probs.get(rid, 1.0)
            
            if "LOAD" in action:
                val = prob * self.value(succ_state, depth-1) + (1-prob) * self.value(state, depth-1)
            elif "POUR" in action:
                reward = self.expected_rewards.get(state[0][0][1], 0) # simplified pos lookup
                # Failure state for pour: robot loses water, but plant doesn't get it
                fail_robots = list(succ_state[0])
                fail_state = (tuple(fail_robots), state[1], state[2], state[3])
                val = prob * (reward + self.value(succ_state, depth-1)) + (1-prob) * self.value(fail_state, depth-1)
            else: # MOVE
                other_moves = [s for a, s in possible if f"({rid})" in a and any(m in a for m in ["UP", "DOWN", "LEFT", "RIGHT"]) and s != succ_state]
                fail_options = other_moves + [state]
                avg_fail = sum(self.value(s, depth-1) for s in fail_options) / len(fail_options)
                val = prob * self.value(succ_state, depth-1) + (1-prob) * avg_fail
            
            best_expected_val = max(best_expected_val, val)
        
        self._value_cache[state_key] = best_expected_val
        return best_expected_val

    def choose_next_action(self, state):
        self._value_cache = {} 
        possible = self.successor(state)
        if not possible: return "RESET"
        
        # KEY ADDITION: If standing on a tap and not full, ALWAYS LOAD.
        # This bypasses the search depth limit for high-capacity robots.
        robots_t = state[0]
        for rid, r_pos, r_load in robots_t:
            capacity = self.capacities[rid]
            if r_load < capacity:
                for t_pos, t_amt in state[2]:
                    if r_pos == t_pos and t_amt > 0:
                        # Check if this robot is the one acting in the possible actions
                        load_action = f"LOAD({rid})"
                        if any(a[0] == load_action for a in possible):
                            return load_action

        # Otherwise, proceed with Expectimax
        best_action, best_val = "RESET", -float('inf')
        for action, succ_state in possible:
            v = self.value(succ_state, self.MAX_DEPTH - 1)
            if v > best_val:
                best_val, best_action = v, action
        return best_action