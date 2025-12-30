import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
import copy
import numpy as np
import re
import sys
import os

# =============================================================================
# חלק 1: הגדרת מחלקת המשחק
# =============================================================================

ACTIONS = {"LOAD", "UP", "RIGHT", "LEFT", "DOWN", "POUR"}

class Game:
    def __init__(self, problem, debug=False):
        self._max_steps = problem["horizon"]
        self._problem = problem
        self._debug = debug
        if self._debug:
            self._history = list()
        self._steps = 0
        self._reward = 0
        self._robot_chosen_action_prob = problem["robot_chosen_action_prob"]
        self._goal_reward = self._problem.get("goal_reward", 0)
        self._plants_reward = problem['plants_reward']
        self._seed = problem["seed"]
        self._done = False
        np.random.seed(self._seed)
        self.rows, self.cols = problem["Size"]
        self.walls  = set(problem.get("Walls", []))
        self.taps   = set(problem.get("Taps", {}).keys())
        self.plants = set(problem.get("Plants", {}).keys())
        
        self._robots = {}
        self._capacities = {}
        for rid, (r, c, carried, capacity) in problem["Robots"].items():
            self._capacities[rid] = capacity
            self._robots[rid] = {"pos": (r, c), "load": carried, "cap": capacity}

        self._plants = {}
        for (r, c), need in problem["Plants"].items():
            self._plants[(r, c)] = {"need": need}

        self._taps = {}
        for (r, c), water in problem["Taps"].items():
            self._taps[(r, c)] = {"water": water}
            
        self._state = self._build_state()
        self._initial_state = self._state
        self.applicable_actions = {}
        self._init_applicable_actions()

    def _init_applicable_actions(self):
        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                if pos in self.walls:
                    self.applicable_actions[pos] = []
                    continue
                actions = []
                if r - 1 >= 0 and (r - 1, c) not in self.walls: actions.append("UP")
                if r + 1 < self.rows and (r + 1, c) not in self.walls: actions.append("DOWN")
                if c - 1 >= 0 and (r, c - 1) not in self.walls: actions.append("LEFT")
                if c + 1 < self.cols and (r, c + 1) not in self.walls: actions.append("RIGHT")
                if pos in self.taps: actions.append("LOAD")
                if pos in self.plants: actions.append("POUR")
                self.applicable_actions[pos] = actions

    def get_current_state(self): return self._state
    def get_max_steps(self): return self._max_steps
    def get_current_reward(self): return self._reward
    def get_problem(self): return self._problem
    def get_current_steps(self): return self._steps
    def get_done(self): return self._done
    def get_capacities(self): return self._capacities

    def parse_robot_action(self, s: str):
        m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*\(\s*([^)]+?)\s*\)\s*", s)
        if not m: raise ValueError(f"Bad action format: {s!r}")
        action = m.group(1).upper()
        if action not in ACTIONS: raise ValueError(f"Unknown action {action!r} in {s!r}")
        robot_id = int(m.group(2).strip())
        return action, robot_id
    
    def submit_next_action(self, chosen_action):
        if self._done: return
        if chosen_action == "RESET":
            self._state = self._initial_state
            self._steps += 1
            if self._steps == self._max_steps: self._done = True
            return

        action_name, robot_id = self.parse_robot_action(chosen_action)
        robots_t, plants_t, taps_t, total_water_need = self._state
        robots_list = list(robots_t); plants_list = list(plants_t); taps_list = list(taps_t)

        robot_idx = None
        for idx, (rid, (rr, cc), l) in enumerate(robots_list):
            if rid == robot_id:
                robot_idx = idx; r, c, load = rr, cc, l; break

        if robot_idx is None: raise ValueError(f"Robot {robot_id} not found")
        robot_pos = (r, c)
        plant_positions = {pos for (pos, need) in plants_t}
        tap_positions = {pos for (pos, water) in taps_t}
        occupied_positions = {(rr, cc) for (rid, (rr, cc), l) in robots_t if rid != robot_id}
        base_applicable = self.applicable_actions[robot_pos]
        dynamic_applicable = []
        for a in base_applicable:
            if a in ("UP", "DOWN", "LEFT", "RIGHT"):
                if a == "UP": target = (r - 1, c)
                elif a == "DOWN": target = (r + 1, c)
                elif a == "LEFT": target = (r, c - 1)
                else: target = (r, c + 1)
                if target in occupied_positions: continue
            dynamic_applicable.append(a)
        
        if (action_name not in dynamic_applicable) or \
           (action_name == "POUR" and (load == 0 or robot_pos not in plant_positions)) or \
           (action_name == "LOAD" and (load >= self._robots[robot_id]["cap"] or robot_pos not in tap_positions)):
            raise ValueError(f"Illegal action {action_name} from {robot_pos} for robot {robot_id}")

        p_success = self._robot_chosen_action_prob[robot_id]
        success = (np.random.rand() < p_success)
        add_reward = 0

        if success:
            if action_name in ("UP", "DOWN", "LEFT", "RIGHT"):
                if action_name == "UP": new_pos = (r - 1, c)
                elif action_name == "DOWN": new_pos = (r + 1, c)
                elif action_name == "LEFT": new_pos = (r, c - 1)
                else: new_pos = (r, c + 1)
                robots_list[robot_idx] = (robot_id, new_pos, load)
            elif action_name == "LOAD":
                tap_idx = next(i for i, (p, w) in enumerate(taps_list) if p == robot_pos)
                pos, water = taps_list[tap_idx]
                if water - 1 > 0: taps_list[tap_idx] = (pos, water - 1)
                else: del taps_list[tap_idx]
                robots_list[robot_idx] = (robot_id, robot_pos, load + 1)
            elif action_name == "POUR":
                plant_idx = next(i for i, (p, n) in enumerate(plants_list) if p == robot_pos)
                pos, need = plants_list[plant_idx]
                if need - 1 > 0: plants_list[plant_idx] = (pos, need - 1)
                else: del plants_list[plant_idx]
                robots_list[robot_idx] = (robot_id, robot_pos, load - 1)
                total_water_need -= 1
                add_reward += np.random.choice(self._plants_reward[robot_pos])
        else:
            if action_name in ("UP", "DOWN", "LEFT", "RIGHT"):
                legal_moves = [a for a in dynamic_applicable if a in ("UP", "DOWN", "LEFT", "RIGHT")]
                alt_moves = [a for a in legal_moves if a != action_name]
                chosen_err = np.random.choice(alt_moves + ["STAY"])
                if chosen_err != "STAY":
                    if chosen_err == "UP": new_pos = (r - 1, c)
                    elif chosen_err == "DOWN": new_pos = (r + 1, c)
                    elif chosen_err == "LEFT": new_pos = (r, c - 1)
                    else: new_pos = (r, c + 1)
                    robots_list[robot_idx] = (robot_id, new_pos, load)
            elif action_name == "POUR":
                robots_list[robot_idx] = (robot_id, robot_pos, load - 1)

        self._reward += add_reward
        if total_water_need == 0:
            self._reward += self._goal_reward
            self._state = self._initial_state
        else:
            self._state = (tuple(robots_list), tuple(plants_list), tuple(taps_list), total_water_need)
        
        self._steps += 1
        if self._steps == self._max_steps: self._done = True

    def _build_state(self):
        robots_t = tuple(sorted([(rid, data["pos"], data["load"]) for rid, data in self._robots.items()]))
        plants_t = tuple(sorted([(pos, data["need"]) for pos, data in self._plants.items()]))
        taps_t = tuple(sorted([(pos, data["water"]) for pos, data in self._taps.items()]))
        total_need = sum(n for _, n in plants_t)
        return (robots_t, plants_t, taps_t, total_need)

# =============================================================================
# חלק 2: הממשק הגרפי (עם חלונית מסלול A* ותיקוני תצוגה)
# =============================================================================

class WaterWorldGUI:
    def __init__(self, root, controller_class, problem_config, loaded_status):
        self.root = root
        self.root.title(f"Water World Debugger | {loaded_status}")
        self.problem_config = problem_config
        self.controller_class = controller_class
        
        self.game = None
        self.controller = None
        self.history = [] 
        self.current_step_index = -1
        self.auto_playing = False
        
        try:
            self.root.state('zoomed') 
        except:
            w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.root.geometry(f"{w}x{h}+0+0")

        self.create_widgets()
        
        self.root.update_idletasks()
        self.calculate_cell_size()
        
        self.reset_game()

    def create_widgets(self):
        # 1. Top Panel (Buttons)
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        style = ttk.Style()
        style.configure("Bold.TButton", font=("Arial", 12, "bold"))
        
        ttk.Button(control_frame, text="⏮ Reset", command=self.reset_game, style="Bold.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="◀ Back", command=self.step_back, style="Bold.TButton").pack(side=tk.LEFT, padx=5)
        self.btn_next = ttk.Button(control_frame, text="Next ▶", command=self.step_forward, style="Bold.TButton")
        self.btn_next.pack(side=tk.LEFT, padx=5)
        self.btn_play = ttk.Button(control_frame, text="▶ Run", command=self.toggle_play, style="Bold.TButton")
        self.btn_play.pack(side=tk.LEFT, padx=5)
        
        self.lbl_step = ttk.Label(control_frame, text="Step: 0", font=("Segoe UI", 16, "bold"))
        self.lbl_step.pack(side=tk.RIGHT, padx=20)
        self.lbl_reward = ttk.Label(control_frame, text="Reward: 0", font=("Segoe UI", 16))
        self.lbl_reward.pack(side=tk.RIGHT, padx=20)
        
        # 2. Info Panel
        info_frame = ttk.Frame(self.root, relief="groove", borderwidth=2)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        lbl_i = ttk.Label(info_frame, text="Intended Action:", font=("Consolas", 12))
        lbl_i.pack(side=tk.LEFT, padx=5)
        self.lbl_intended = ttk.Label(info_frame, text="-", font=("Consolas", 14, "bold"), foreground="blue")
        self.lbl_intended.pack(side=tk.LEFT, padx=10)
        
        lbl_o = ttk.Label(info_frame, text="Actual Outcome:", font=("Consolas", 12))
        lbl_o.pack(side=tk.LEFT, padx=5)
        self.lbl_outcome = ttk.Label(info_frame, text="-", font=("Consolas", 14, "bold"))
        self.lbl_outcome.pack(side=tk.LEFT, padx=10)

        # 3. Main Split
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas_frame = tk.Frame(main_paned, bg="#333")
        main_paned.add(self.canvas_frame, weight=3)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(anchor="center", expand=True)
        
        # A* Path Frame
        path_frame = ttk.LabelFrame(main_paned, text="Controller Plan", padding=5)
        main_paned.add(path_frame, weight=1)
        
        self.path_listbox = tk.Listbox(path_frame, font=("Consolas", 11), bg="#fdfdfd")
        self.path_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scrollbar = ttk.Scrollbar(path_frame, orient="vertical", command=self.path_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.path_listbox.config(yscrollcommand=scrollbar.set)

        self.root.bind("<Configure>", self.on_resize)

    def calculate_cell_size(self):
        avail_w = self.canvas_frame.winfo_width()
        avail_h = self.canvas_frame.winfo_height()
        if avail_w <= 1: avail_w = self.root.winfo_screenwidth() // 2
        if avail_h <= 1: avail_h = self.root.winfo_screenheight() - 200
        
        rows, cols = self.problem_config["Size"]
        self.cell_size = min((avail_w - 20) // cols, (avail_h - 20) // rows)
        self.canvas.config(width=self.cell_size * cols, height=self.cell_size * rows)
        
    def on_resize(self, event):
        if hasattr(self, 'last_resize_time'):
            if time.time() - self.last_resize_time < 0.1: return
        self.last_resize_time = time.time()
        self.calculate_cell_size()
        if self.history: self.draw_state()

    def _get_controller_path(self):
        """Extracts the path from the controller regardless of variable naming"""
        if not self.controller: return [], 0
        
        # Try both naming conventions you used
        path = getattr(self.controller, 'plan', [])
        idx = getattr(self.controller, 'current_path_index', 0)
        return path, idx

    def reset_game(self):
        self.auto_playing = False
        self.btn_play.config(text="▶ Run")
        self.game = Game(self.problem_config)
        self.controller = self.controller_class(self.game)
        
        # FIX: Get path immediately after init!
        path, idx = self._get_controller_path()
        
        initial_snapshot = {
            'state': self.game.get_current_state(),
            'reward': 0, 'steps': 0,
            'intended': "Start", 'outcome': "-", 'done': False,
            'path': copy.copy(path), 
            'path_idx': idx
        }
        self.history = [initial_snapshot]
        self.current_step_index = 0
        self.draw_state()

    def update_path_display(self, path, current_index):
        self.path_listbox.delete(0, tk.END)
        if not path:
            self.path_listbox.insert(tk.END, "(Computing or Empty)")
            return

        for i, action in enumerate(path):
            prefix = "   "
            if i < current_index:
                prefix = "✓ "
            elif i == current_index:
                prefix = "➜ "
            
            self.path_listbox.insert(tk.END, f"{prefix}{action}")
            
            if i < current_index:
                self.path_listbox.itemconfig(i, {'fg': 'gray'})
            elif i == current_index:
                self.path_listbox.itemconfig(i, {'fg': 'blue', 'bg': '#e6f7ff'})
            else:
                self.path_listbox.itemconfig(i, {'fg': 'black'})
        
        self.path_listbox.see(current_index)

    def analyze_outcome(self, prev_state, new_state, action_str):
        if action_str == "RESET": return "Game Reset", "black"
        try:
            match = re.match(r"([A-Z]+)\s*\((\d+)\)", action_str)
            if not match: return "Invalid Format", "gray"
            act, rid = match.groups(); rid = int(rid)
            
            prev_robots = {r[0]: {'pos': r[1], 'load': r[2]} for r in prev_state[0]}
            new_robots = {r[0]: {'pos': r[1], 'load': r[2]} for r in new_state[0]}
            
            p_r = prev_robots[rid]; n_r = new_robots[rid]
            
            if act in ["UP", "DOWN", "LEFT", "RIGHT"]:
                if p_r['pos'] == n_r['pos']: return "FAILED (Stayed)", "red"
                pr, pc = p_r['pos']; nr, nc = n_r['pos']
                dr, dc = nr - pr, nc - pc
                expected = {"UP": (-1,0), "DOWN": (1,0), "LEFT": (0,-1), "RIGHT": (0,1)}
                if (dr, dc) == expected[act]: return "SUCCESS", "green"
                else:
                    slip_map = {(-1,0):"UP", (1,0):"DOWN", (0,-1):"LEFT", (0,1):"RIGHT"}
                    return f"SLIPPED {slip_map.get((dr, dc), '?')}!", "red"
            
            elif act == "LOAD":
                if n_r['load'] > p_r['load']: return "SUCCESS", "green"
                return "FAILED", "red"
                
            elif act == "POUR":
                plant_pos = p_r['pos']
                prev_plants = {p[0]: p[1] for p in prev_state[1]}
                new_plants = {p[0]: p[1] for p in new_state[1]}
                p_need = prev_plants.get(plant_pos, 0)
                n_need = new_plants.get(plant_pos, 0)
                
                if n_need < p_need or (p_need > 0 and plant_pos not in new_plants): return "SUCCESS", "green"
                elif n_r['load'] < p_r['load']: return "SPILL!", "red"
                else: return "FAILED", "red"
        except: return "Unknown", "gray"
        return "Done", "black"

    def step_forward(self):
        if self.current_step_index < len(self.history) - 1:
            self.current_step_index += 1
            self.draw_state()
            return

        if self.game.get_done(): return

        prev_state = copy.deepcopy(self.game.get_current_state())
        try:
            # 1. Get Intended Action
            action = self.controller.choose_next_action(prev_state)
            
            # Extract Path info
            path, path_idx = self._get_controller_path()
            
            # 2. Execute
            self.game.submit_next_action(action)
            new_state = self.game.get_current_state()
            
            # 3. Analyze
            outcome_text, outcome_color = self.analyze_outcome(prev_state, new_state, action)
            
            snapshot = {
                'state': new_state,
                'reward': self.game.get_current_reward(),
                'steps': self.game.get_current_steps(),
                'intended': action,
                'outcome': outcome_text,
                'outcome_color': outcome_color,
                'done': self.game.get_done(),
                'path': copy.copy(path),
                'path_idx': path_idx
            }
            self.history.append(snapshot)
            self.current_step_index += 1
            self.draw_state()
            
        except Exception as e:
            messagebox.showerror("Error", f"Controller Error:\n{e}")
            self.auto_playing = False

    def step_back(self):
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.draw_state()

    def toggle_play(self):
        if self.auto_playing:
            self.auto_playing = False
            self.btn_play.config(text="▶ Run")
        else:
            self.auto_playing = True
            self.btn_play.config(text="⏸ Pause")
            threading.Thread(target=self.play_loop, daemon=True).start()

    def play_loop(self):
        while self.auto_playing and not self.game.get_done():
            self.root.after(0, self.step_forward)
            time.sleep(0.1)
        if self.game.get_done():
            self.auto_playing = False
            self.root.after(0, lambda: self.btn_play.config(text="▶ Run"))

    def draw_state(self):
        self.canvas.delete("all")
        snap = self.history[self.current_step_index]
        state = snap['state']
        
        self.lbl_step.config(text=f"Step: {snap['steps']} / {self.problem_config['horizon']}")
        self.lbl_reward.config(text=f"Reward: {snap['reward']}")
        self.lbl_intended.config(text=snap['intended'])
        self.lbl_outcome.config(text=snap['outcome'], foreground=snap.get('outcome_color', 'black'))
        
        self.update_path_display(snap.get('path', []), snap.get('path_idx', 0))
        
        rows, cols = self.problem_config["Size"]
        robots_t, plants_t, taps_t, _ = state
        robots = {r[0]: {'pos': r[1], 'load': r[2]} for r in robots_t}
        plants = {p[0]: p[1] for p in plants_t}
        taps = {t[0]: t[1] for t in taps_t}
        walls = set(self.problem_config["Walls"])
        capacities = self.game.get_capacities()

        base_font = max(8, int(self.cell_size / 6))
        bold_font = ("Arial", base_font, "bold")
        detail_font = ("Arial", int(base_font * 1.2), "bold")
        
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                cx, cy = (x1+x2)/2, (y1+y2)/2
                pos = (r, c)
                
                fill_color = "#222" if pos in walls else "#f0f0f0"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="#ccc")
                if pos in walls: continue
                
                if pos in taps:
                    pad = self.cell_size * 0.05
                    self.canvas.create_oval(x1+pad, y1+pad, x2-pad, y2-pad, fill="#e6f7ff", outline="#1890ff", width=2)
                    self.canvas.create_text(cx, cy, text=f"💧\n{taps[pos]}", fill="#0050b3", font=detail_font, justify="center")

                if pos in plants:
                    pad = self.cell_size * 0.1
                    self.canvas.create_rectangle(x1+pad, y1+pad, x2-pad, y2-pad, fill="#f6ffed", outline="#52c41a", width=2)
                    self.canvas.create_text(cx, cy, text=f"🌱\n{plants[pos]}", fill="#237804", font=detail_font, justify="center")

                cell_robots = [rid for rid, data in robots.items() if data['pos'] == pos]
                if cell_robots:
                    for rid in cell_robots:
                        r_pad = self.cell_size * 0.15
                        self.canvas.create_oval(x1+r_pad+2, y1+r_pad+2, x2-r_pad+2, y2-r_pad+2, fill="#aaa", outline="")
                        self.canvas.create_oval(x1+r_pad, y1+r_pad, x2-r_pad, y2-r_pad, fill="#ffbf00", outline="#874d00", width=3)
                        load = robots[rid]['load']
                        cap = capacities.get(rid, "?")
                        self.canvas.create_text(cx, cy - self.cell_size*0.15, text=f"R{rid}", font=bold_font, fill="#5b3200")
                        self.canvas.create_text(cx, cy + self.cell_size*0.15, text=f"{load}/{cap}", font=detail_font, fill="black")

# =============================================================================
# חלק 3: טעינת הקונפיגורציה
# =============================================================================


problem_pdf = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.9,
    },
    "goal_reward": 10,
    "plants_reward": {
        (0, 2): [1, 2, 3, 4],
        (2, 0): [1, 2, 3, 4],
    },
    "seed": 45,
    "horizon": 30,
}

problem_pdf2 = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.9,
        11: 0.8,
    },
    "goal_reward": 12,
    "plants_reward": {
        (0, 2): [1, 3, 5, 7],
        (2, 0): [1, 2, 3, 4],
    },
    "seed": 45,
    "horizon": 35,
}

problem_pdf3 = {
    "Size": (3, 3),
    "Walls": {(0, 1), (2, 1)},
    "Taps": {(1, 1): 6},
    "Plants": {(2, 0): 2, (0, 2): 3},
    "Robots": {10: (1, 0, 0, 2), 11: (1, 2, 0, 2)},
    "robot_chosen_action_prob": {
        10: 0.7,
        11: 0.6,
    },
    "goal_reward": 30,
    "plants_reward": {
        (0, 2): [1, 2, 3, 4],
        (2, 0): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new1_version1 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
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
    "robot_chosen_action_prob": {
        10: 0.9,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new1_version2 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
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
    "robot_chosen_action_prob": {
        10: 0.6,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 70,
}

problem_new1_version3 = {
    "Size": (5, 6),
    "Walls": {
        # block some middle cells to create a kind of corridor
        (1, 2),
        (1, 3),
        (3, 2),
        (3, 3),
    },
    "Taps": {
        (2, 2): 12,
    },
    "Plants": {
        (0, 1): 2,
        (4, 5): 6,
    },
    "Robots": {
        10: (2, 1, 0, 6),
        11: (2, 4, 0, 3),
    },
    "robot_chosen_action_prob": {
        10: 0.6,
        11: 0.95,
    },
    "goal_reward": 30,
    "plants_reward": {
        (4, 5): [1, 2, 3, 4],
        (0, 1): [10, 11, 12, 13],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new2_version1 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 30,
}


problem_new2_version2 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 70,
}

problem_new2_version3 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 20,
    "plants_reward": {
        (0, 0): [5, 7, 9],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new2_version4 = {
    "Size": (5, 6),
    "Walls": {
        # corridor shifted up
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    },
    "Taps": {
        (1, 2): 10,  # upper tap
        (3, 3): 10,  # lower tap
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (4, 5): 5,  # bottom-right
    },
    "Robots": {
        10: (1, 1, 0, 5),  # near upper tap, cap 3
        11: (3, 4, 0, 4),  # near lower tap, cap 2
    },
    "robot_chosen_action_prob": {
        10: 0.7,
        11: 0.95,
    },
    "goal_reward": 18,
    "plants_reward": {
        (0, 0): [5, 7],
        (4, 5): [5, 7],
    },
    "seed": 45,
    "horizon": 40,
}

problem_new3_version1 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,  # upper-right corrido
        (9, 0): 10,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.95,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 30,
}

problem_new3_version2 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 10,  # upper-right corrido
        (9, 0): 10,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.8,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 50,
}

problem_new3_version3 = {
    "Size": (10, 4),
    "Walls": {
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (4, 2),
        (4, 3),
        (6, 2),
        (6, 3),
    },
    # Tap on the left side, with enough water
    "Taps": {
        (5, 3): 20,
    },
    # Plants on the far right, all need water
    "Plants": {
        (0, 0): 5,  # upper-right corrido
        (9, 0): 5,
    },
    # Single robot, small capacity → many long trips through the maze
    "Robots": {
        10: (2, 0, 0, 2),  # bottom-left area near the tap side
        11: (7, 0, 0, 20),  # bottom-left area near the tap side
    },
    "robot_chosen_action_prob": {
        10: 0.95,
        11: 0.0001,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (9, 0): [1, 3],
    },
    "seed": 45,
    "horizon": 70,
}
# reset ?
problem_new4_version1 = {
    "Size": (10, 10),
    "Walls": set(),  # completely open grid
    "Taps": {
        (8, 8): 24,
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (0, 9): 5,  # top-right
        (9, 0): 5,  # bottom-left
        (9, 9): 5,  # bottom-right
        # total need = 20
    },
    "Robots": {
        10: (8, 9, 0, 5),
    },
    "robot_chosen_action_prob": {
        10: 0.95,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (0, 9): [1, 3],
        (9, 0): [1, 3],
        (9, 9): [1, 3],
    },
    "seed": 45,
    "horizon": 70,
}

# reset ?
problem_new4_version2 = {
    "Size": (10, 10),
    "Walls": set(),  # completely open grid
    "Taps": {
        (8, 8): 24,
    },
    "Plants": {
        (0, 0): 5,  # top-left
        (0, 9): 5,  # top-right
        (9, 0): 5,  # bottom-left
        (9, 9): 5,  # bottom-right
        # total need = 20
    },
    "Robots": {
        10: (8, 9, 0, 5),
    },
    "robot_chosen_action_prob": {
        10: 0.85,
    },
    "goal_reward": 9,
    "plants_reward": {
        (0, 0): [1, 3],
        (0, 9): [1, 3],
        (9, 0): [1, 3],
        (9, 9): [1, 3],
    },
    "seed": 45,
    "horizon": 40,
}

ROBLEM_CONFIG = {
    "Size": (4, 4),
    "Walls": [(1, 1), (2, 2)],
    "Robots": {0: (0, 0, 0, 2), 1: (3, 3, 0, 2)},
    "Plants": {(0, 3): 5, (3, 0): 3},
    "Taps": {(0, 2): 10, (2, 0): 10},
    "robot_chosen_action_prob": {0: 0.95, 1: 0.8},
    "plants_reward": {(0, 3): [10, 20], (3, 0): [5, 5]},
    "horizon": 50, "seed": 42, "goal_reward": 100
}

PROBLEM_CONFIG = problem_pdf

if __name__ == "__main__":
    status = "Loaded: Dummy (ex2.py not found)"
    try:
        import ex2
        ControllerClass = ex2.Controller
        status = "Loaded: ex2.py"
        print(status)
    except ImportError:
        print("Could not find 'ex2.py'. Using Dummy.")
        class DummyController:
            def __init__(self, game): self.game = game
            def choose_next_action(self, state): return "RESET"
        ControllerClass = DummyController

    root = tk.Tk()
    app = WaterWorldGUI(root, ControllerClass, PROBLEM_CONFIG, status)
    root.mainloop()
