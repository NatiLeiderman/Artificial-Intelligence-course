id = ["331010090"]

"""
AI Usage: firstly I used Gemini to implement the program, but it did very poorly so I rewrote the whole program by myself
and it run 20x faster. Then after a few more optemizations which I implented by myself I got to very satisfuying running times.
I did use AI to check my code a few times, mostly to clarify my heurstics are admisible (although is was mistaken on that part
a few times)
"""

import search
import utils

def _manhattan_dist(pos1, pos2):
    # Calculating the manhetten distances
    r1, c1 = pos1
    r2, c2 = pos2
    return abs(r1 - r2) + abs(c1 - c2)

class WateringProblem(search.Problem):
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
        search.Problem.__init__(self, initial_state)
        
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
            if robot_pos in self.tap_index_map and carry < capacity:
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
                    min_cost_to_service_plant = utils.infinity
                    
                    for r_row, r_col, carry, _ in robots_state:
                        robot_pos = (r_row, r_col)
                        cost = utils.infinity
                        
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
        min_next_action_distance = utils.infinity
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
            min_active_tap_to_plant_dist_for_dry_robot = utils.infinity
            if active_taps_positions and unwatered_plants_positions:
                min_active_tap_to_plant_dist_for_dry_robot = min(
                    self._tap_plant_distances[(tap_pos, p_pos)]
                    for tap_pos in active_taps_positions
                    for p_pos in unwatered_plants_positions
                )
            
            for r_row, r_col, carry, _ in robots_state:
                robot_pos = (r_row, r_col)
                current_robot_best_path = utils.infinity

                if carry > 0 and unwatered_plants_positions:
                    current_robot_best_path = min(
                        _manhattan_dist(robot_pos, p_pos) 
                        for p_pos in unwatered_plants_positions
                    )
                elif active_taps_positions:
                    min_robot_to_tap_dist = min(
                        _manhattan_dist(robot_pos, tap_pos) for tap_pos in active_taps_positions
                    )
                    
                    if min_active_tap_to_plant_dist_for_dry_robot != utils.infinity:
                        current_robot_best_path = min_robot_to_tap_dist + min_active_tap_to_plant_dist_for_dry_robot
                
                min_next_action_distance = min(min_next_action_distance, current_robot_best_path)

        # Weighted calculation
        load_component = total_loads_needed * 10000
        pour_component = total_pours_needed * 100
        distance_component = min_next_action_distance if min_next_action_distance != utils.infinity else 0

        h_value = load_component + pour_component + distance_component
        return h_value

def create_watering_problem(game):
    return WateringProblem(game)