import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
import copy
import numpy as np
import re
import sys
import os

# Try to import the game and controller
try:
    import ext_plant
    import ex2
except ImportError as e:
    print(f"Error: Could not import required files. Make sure 'ext_plant.py' and 'ex2.py' are in the same folder.\nDetails: {e}")
    sys.exit(1)

# =============================================================================
#  CONFIGURATION (Sample Problem)
# =============================================================================
PROBLEM_CONFIG = {
    "Size": (10, 10),
    "Walls": set(), # Open grid
    "Taps": {(8, 8): 24},
    "Plants": {
        (0, 0): 5, 
        (0, 9): 5, 
        (9, 0): 5, 
        (9, 9): 5
    },
    "Robots": {
        10: (8, 9, 0, 5),   
    },
    "robot_chosen_action_prob": {10: 0.95},
    "goal_reward": 100,
    "plants_reward": {
        (0, 0): [10], (0, 9): [10], (9, 0): [10], (9, 9): [10]
    },
    "seed": 42,
    "horizon": 100,
}

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

PROBLEM_CONFIG = problem_pdf3
# =============================================================================
#  THE VISUALIZER CLASS
# =============================================================================
class WaterWorldGUI:
    def __init__(self, root, controller_class, problem_config):
        self.root = root
        self.root.title("Water World AI Visualizer")
        self.problem = problem_config
        self.ControllerClass = controller_class
        
        self.game = None
        self.controller = None
        self.history = [] 
        self.current_step_index = -1
        self.auto_playing = False
        self.cell_size = 60
        
        # Setup Window
        w, h = 1200, 800
        self.root.geometry(f"{w}x{h}")
        
        self.setup_ui()
        self.reset_game()

    def setup_ui(self):
        # --- TOP TOOLBAR ---
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(fill=tk.X, side=tk.TOP)
        
        ttk.Button(toolbar, text="⏮ Reset", command=self.reset_game).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="▶ Run/Pause", command=self.toggle_play).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Step ➜", command=self.step_forward).pack(side=tk.LEFT, padx=2)
        
        self.lbl_status = ttk.Label(toolbar, text="Ready", font=("Arial", 12, "bold"))
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        # --- MAIN CONTENT ---
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Canvas
        self.canvas_frame = ttk.Frame(paned)
        paned.add(self.canvas_frame, weight=3)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#2b2b2b")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right: Info & Plan
        right_panel = ttk.Frame(paned, padding=5)
        paned.add(right_panel, weight=1)
        
        # Info Box
        info_frame = ttk.LabelFrame(right_panel, text="Step Info")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_action = ttk.Label(info_frame, text="Action: -", font=("Consolas", 11))
        self.lbl_action.pack(anchor="w")
        self.lbl_result = ttk.Label(info_frame, text="Result: -", font=("Consolas", 11))
        self.lbl_result.pack(anchor="w")
        
        # Plan Box
        plan_frame = ttk.LabelFrame(right_panel, text="Current Plan (A*)")
        plan_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.lst_plan = tk.Listbox(plan_frame, font=("Consolas", 10), bg="#f0f0f0")
        self.lst_plan.pack(fill=tk.BOTH, expand=True)

    def reset_game(self):
        self.auto_playing = False
        self.game = ext_plant.Game(self.problem)
        self.controller = self.ControllerClass(self.game)
        
        # Snapshot initial state
        self.history = []
        self.save_snapshot("Start", "None")
        self.current_step_index = 0
        self.draw_current_state()

    def save_snapshot(self, action, result):
        current_plan = getattr(self.controller, 'plan', [])
        snapshot = {
            'state': copy.deepcopy(self.game.get_current_state()),
            'step': self.game.get_current_steps(),
            'reward': self.game.get_current_reward(),
            'action': action,
            'result': result,
            'plan': copy.copy(current_plan)
        }
        self.history.append(snapshot)

    def step_forward(self):
        if self.game.get_done():
            self.lbl_status.config(text="Game Over!")
            return

        prev_state = copy.deepcopy(self.game.get_current_state())
        
        try:
            action = self.controller.choose_next_action(prev_state)
        except Exception as e:
            messagebox.showerror("Controller Error", str(e))
            return

        self.game.submit_next_action(action)
        
        new_state = self.game.get_current_state()
        result_str = self.analyze_move(prev_state, new_state, action)
        
        self.save_snapshot(action, result_str)
        self.current_step_index += 1
        self.draw_current_state()

    def analyze_move(self, prev, curr, action_str):
        if action_str == "RESET": return "Reset Game"
        
        try:
            parts = action_str.split('(')
            act_type = parts[0].strip()
            rid = int(parts[1].replace(')', '').strip())
        except: return "Unknown"

        def get_bot(s): 
            return next((r for r in s[0] if r[0] == rid), None)

        p_bot = get_bot(prev)
        c_bot = get_bot(curr)
        
        if not p_bot or not c_bot: return "Error"

        if act_type in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if p_bot[1] == c_bot[1]: return "FAILED (Blocked/Stayed)"
            
            pr, pc = p_bot[1]
            cr, cc = c_bot[1]
            dr, dc = cr-pr, cc-pc
            
            expected = {"UP":(-1,0), "DOWN":(1,0), "LEFT":(0,-1), "RIGHT":(0,1)}
            if (dr, dc) == expected[act_type]:
                return "SUCCESS"
            else:
                return f"SLIPPED! ({dr},{dc})"
        
        elif act_type == "LOAD":
            if c_bot[2] > p_bot[2]: return "SUCCESS (Loaded)"
            return "FAILED (No water?)"
            
        elif act_type == "POUR":
            plant_pos = p_bot[1]
            p_plant = next((p for p in prev[1] if p[0] == plant_pos), None)
            c_plant = next((p for p in curr[1] if p[0] == plant_pos), None)
            
            if p_plant and (not c_plant or c_plant[1] < p_plant[1]):
                return "SUCCESS (Watered)"
            return "FAILED (Spilled)"
            
        return "Done"

    def draw_current_state(self):
        self.canvas.delete("all")
        snap = self.history[self.current_step_index]
        state = snap['state']
        
        # --- STATUS & INFO ---
        max_steps = self.problem.get('horizon', '?')
        self.lbl_status.config(text=f"Step: {snap['step']} / {max_steps} | Reward: {snap['reward']}")
        self.lbl_action.config(text=f"Action: {snap['action']}", foreground="blue")
        
        res_color = "green" if "SUCCESS" in snap['result'] else "red"
        if "Start" in snap['result']: res_color = "black"
        self.lbl_result.config(text=f"Result: {snap['result']}", foreground=res_color)
        
        # --- UPDATE PLAN ---
        self.lst_plan.delete(0, tk.END)
        for i, step in enumerate(snap['plan']):
            self.lst_plan.insert(tk.END, f"{i+1}. {step}")
            
        # --- DRAW GRID ---
        rows, cols = self.problem['Size']
        cw = self.canvas.winfo_width() / cols
        ch = self.canvas.winfo_height() / rows
        size = min(cw, ch, 80)
        
        offset_x = (self.canvas.winfo_width() - (cols * size)) / 2
        offset_y = (self.canvas.winfo_height() - (rows * size)) / 2

        walls = self.problem['Walls']
        taps = {t[0]: t[1] for t in state[2]} # (r,c): amt
        plants = {p[0]: p[1] for p in state[1]} # (r,c): need
        robots = {r[0]: {'pos': r[1], 'load': r[2]} for r in state[0]}
        
        # Get capacities from Game object
        capacities = self.game.get_capacities()

        for r in range(rows):
            for c in range(cols):
                x1 = offset_x + c * size
                y1 = offset_y + r * size
                x2, y2 = x1 + size, y1 + size
                
                # Base Cell
                color = "#444" if (r,c) in walls else "#ddd"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="white")
                
                # Tap
                if (r,c) in taps:
                    pad = size * 0.1
                    self.canvas.create_oval(x1+pad, y1+pad, x2-pad, y2-pad, fill="#aaf", outline="blue", width=2)
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=f"💧{taps[(r,c)]}")

                # Plant
                if (r,c) in plants:
                    pad = size * 0.2
                    self.canvas.create_rectangle(x1+pad, y1+pad, x2-pad, y2-pad, fill="#afa", outline="green", width=2)
                    self.canvas.create_text((x1+x2)/2, (y1+y2)/2, text=f"🌱{plants[(r,c)]}")

        # Draw Robots
        for rid, data in robots.items():
            rr, rc = data['pos']
            x1 = offset_x + rc * size + (size*0.25)
            y1 = offset_y + rr * size + (size*0.25)
            x2 = x1 + (size*0.5)
            y2 = y1 + (size*0.5)
            
            # Draw Robot Body
            self.canvas.create_oval(x1, y1, x2, y2, fill="orange", outline="black", width=2)
            
            # Robot ID
            self.canvas.create_text((x1+x2)/2, y1-10, text=f"R{rid}", fill="black", font=("Arial", 10, "bold"))
            
            # Load / Capacity
            cap = capacities.get(rid, "?")
            self.canvas.create_text((x1+x2)/2, y2+10, text=f"{data['load']}/{cap}", fill="black", font=("Arial", 9, "bold"))

    def toggle_play(self):
        if self.auto_playing:
            self.auto_playing = False
        else:
            self.auto_playing = True
            threading.Thread(target=self.play_loop, daemon=True).start()

    def play_loop(self):
        while self.auto_playing and not self.game.get_done():
            self.root.after(0, self.step_forward)
            time.sleep(0.1) 
        self.auto_playing = False

# =============================================================================
#  MAIN
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    if 'ex2' in sys.modules:
        app = WaterWorldGUI(root, ex2.Controller, PROBLEM_CONFIG)
        root.mainloop()
    else:
        print("Failed to load ex2.Controller")
