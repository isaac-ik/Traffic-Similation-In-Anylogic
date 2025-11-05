import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class MPCSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("MPC Traffic Control Simulator")
        self.root.geometry("1200x800")
        
        # Simulation state
        self.running = False
        self.paused = False
        self.current_step = 0
        self.sim_thread = None
        self.param_lock = threading.Lock()
        
        # Default parameters
        self.dt = 1.0
        self.lambda_nominal = 10.0
        self.mu = 15.0
        self.N = 10
        self.q0 = 20.0
        self.q_ref_1 = 5.0
        self.q_ref_2 = 8.0
        self.q_ref_switch = 60
        
        # History storage (rolling window)
        self.max_history = 300
        self.Qhist = []
        self.Uhist = []
        self.time_hist = []
        self.q_ref_hist = []
        
        # Setup UI
        self.setup_ui()
        self.initialize_simulation()
        
    def setup_ui(self):
        # Control panel (left side)
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Parameters
        ttk.Label(control_frame, text="Simulation Parameters", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)
        ttk.Label(control_frame, text="(Adjust during simulation)", font=('Arial', 9, 'italic')).grid(row=1, column=0, columnspan=2)
        
        params = [
            ("Target Queue 1:", "q_ref_1", 0, 20, 5.0),
            ("Target Queue 2:", "q_ref_2", 0, 20, 8.0),
            ("Switch Time (min):", "q_ref_switch", 0, 120, 60),
            ("Inflow Rate (λ):", "lambda_nominal", 1, 20, 10.0),
            ("Outflow Rate (μ):", "mu", 1, 30, 15.0),
            ("Horizon (N):", "N", 1, 30, 10),
        ]
        
        self.sliders = {}
        self.labels = {}
        
        for i, (label, var, min_val, max_val, default) in enumerate(params):
            row = i + 2
            ttk.Label(control_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
            
            value_label = ttk.Label(control_frame, text=f"{default:.1f}")
            value_label.grid(row=row, column=2, sticky=tk.W, padx=5)
            self.labels[var] = value_label
            
            slider = ttk.Scale(control_frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                             command=lambda v, var=var, lbl=value_label: self.update_param(var, v, lbl))
            slider.set(default)
            slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
            self.sliders[var] = slider
        
        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=len(params)+2, column=0, columnspan=3, pady=20)
        
        self.start_btn = ttk.Button(button_frame, text="Start", command=self.start_simulation)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.pause_btn = ttk.Button(button_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=1, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, padx=5)
        
        self.reset_btn = ttk.Button(button_frame, text="Reset", command=self.reset_simulation)
        self.reset_btn.grid(row=0, column=3, padx=5)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Ready", font=('Arial', 10))
        self.status_label.grid(row=len(params)+3, column=0, columnspan=3, pady=10)
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.grid(row=len(params)+4, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.time_label = ttk.Label(stats_frame, text="Time: 0 min")
        self.time_label.grid(row=0, column=0, sticky=tk.W)
        
        self.current_queue_label = ttk.Label(stats_frame, text="Current Queue: --")
        self.current_queue_label.grid(row=1, column=0, sticky=tk.W)
        
        self.avg_queue_label = ttk.Label(stats_frame, text="Avg Queue (60min): --")
        self.avg_queue_label.grid(row=2, column=0, sticky=tk.W)
        
        self.max_queue_label = ttk.Label(stats_frame, text="Max Queue (60min): --")
        self.max_queue_label.grid(row=3, column=0, sticky=tk.W)
        
        self.current_target_label = ttk.Label(stats_frame, text="Current Target: --")
        self.current_target_label.grid(row=4, column=0, sticky=tk.W)
        
        # Plots (right side)
        plot_frame = ttk.Frame(self.root, padding="10")
        plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.fig = Figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def update_param(self, var, value, label):
        with self.param_lock:
            val = float(value)
            if var in ['N', 'q_ref_switch']:
                val = int(val)
            setattr(self, var, val)
            label.config(text=f"{val:.1f}" if isinstance(val, float) else f"{val}")
            
            # Update matrices if flow rates change
            if var in ['lambda_nominal', 'mu']:
                self.B = np.array([[self.dt * self.lambda_nominal, -self.dt * self.mu]])
        
    def initialize_simulation(self):
        self.A = np.array([[1.0]])
        self.B = np.array([[self.dt * self.lambda_nominal, -self.dt * self.mu]])
        self.Q = np.array([[1.0]])
        self.R = np.diag([0.1, 0.1])
        
        self.Qhist = [self.q0]
        self.Uhist = []
        self.time_hist = [0]
        self.q_ref_hist = [self.q_ref_1]
        self.current_step = 0
        self.current_q = self.q0
        
    def get_current_target(self):
        """Get current target based on elapsed time and switch parameter"""
        with self.param_lock:
            if self.current_step < self.q_ref_switch:
                return self.q_ref_1
            else:
                return self.q_ref_2
        
    def finite_horizon_lqr_gains(self):
        with self.param_lock:
            N = self.N
            Q = self.Q.copy()
            R = self.R.copy()
            A = self.A.copy()
            B = self.B.copy()
        
        P = [None] * (N + 1)
        K = [None] * N
        P[N] = Q
        
        for k in range(N-1, -1, -1):
            S = R + B.T @ P[k+1] @ B
            Kk = np.linalg.solve(S, B.T @ P[k+1] @ A)
            K[k] = Kk
            P[k] = Q + A.T @ P[k+1] @ (A - B @ Kk)
        
        return K, P
    
    def simulation_step(self):
        # Get current parameters (thread-safe)
        with self.param_lock:
            A = self.A.copy()
            B = self.B.copy()
            lambda_nom = self.lambda_nominal
            mu_val = self.mu
        
        # Calculate LQR gains with current parameters
        K_seq, _ = self.finite_horizon_lqr_gains()
        
        # Get current target
        q_ref_current = self.get_current_target()
        
        # Compute control
        q_err = self.current_q - q_ref_current
        K0 = K_seq[0]
        u = -K0 @ np.array([[q_err]])
        u = u.ravel()
        u = np.clip(u, 0.0, 1.0)
        
        # Simulate dynamics with noise
        w = np.random.normal(scale=0.2)
        q_next = (A @ np.array([[self.current_q]]) + (B @ u.reshape(2,1))).item() + w
        q_next = max(q_next, 0.0)
        
        # Store history (rolling window)
        self.current_q = q_next
        self.current_step += 1
        
        self.Qhist.append(q_next)
        self.Uhist.append(u)
        self.time_hist.append(self.current_step)
        self.q_ref_hist.append(q_ref_current)
        
        # Keep only recent history
        if len(self.Qhist) > self.max_history:
            self.Qhist.pop(0)
            self.Uhist.pop(0)
            self.time_hist.pop(0)
            self.q_ref_hist.pop(0)
        
        self.update_plots()
        
    def update_plots(self):
        if len(self.Qhist) < 2:
            return
        
        time = np.array(self.time_hist)
        queue = np.array(self.Qhist)
        target = np.array(self.q_ref_hist)
        
        self.ax1.clear()
        self.ax1.plot(time, queue, 'b-', linewidth=2, label='Queue')
        self.ax1.plot(time, target, 'r--', linewidth=2, label='Target')
        self.ax1.set_ylabel('Queued Cars')
        self.ax1.set_xlabel('Time (minutes)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Queue Evolution (Live)')
        
        if len(self.Uhist) > 0:
            uhist = np.array(self.Uhist)
            time_ctrl = time[:-1]  # Control is one step behind
            
            self.ax2.clear()
            self.ax2.step(time_ctrl, uhist[:, 0], where='post', label='Metering (u1)', linewidth=2)
            self.ax2.step(time_ctrl, uhist[:, 1], where='post', label='Green (u2)', linewidth=2)
            self.ax2.set_ylabel('Control (fraction)')
            self.ax2.set_xlabel('Time (minutes)')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(-0.1, 1.1)
            self.ax2.set_title('Control Inputs (Live)')
            
            with self.param_lock:
                lambda_nom = self.lambda_nominal
                mu_val = self.mu
            
            inflow = uhist[:, 0] * lambda_nom
            outflow = uhist[:, 1] * mu_val
            
            self.ax3.clear()
            self.ax3.step(time_ctrl, inflow, where='post', label='Inflow', linewidth=2)
            self.ax3.step(time_ctrl, outflow, where='post', label='Outflow', linewidth=2)
            self.ax3.set_ylabel('Cars per minute')
            self.ax3.set_xlabel('Time (minutes)')
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)
            self.ax3.set_title('Flow Rates (Live)')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update statistics (last 60 steps or all if less)
        window = min(60, len(queue))
        recent_queue = queue[-window:]
        
        avg_q = np.mean(recent_queue)
        max_q = np.max(recent_queue)
        current_q = queue[-1]
        current_target = target[-1]
        
        self.root.after(0, self.update_stats, current_q, avg_q, max_q, current_target)
        
    def update_stats(self, current_q, avg_q, max_q, current_target):
        self.time_label.config(text=f"Time: {self.current_step} min")
        self.current_queue_label.config(text=f"Current Queue: {current_q:.2f} cars")
        self.avg_queue_label.config(text=f"Avg Queue (60min): {avg_q:.2f} cars")
        self.max_queue_label.config(text=f"Max Queue (60min): {max_q:.2f} cars")
        self.current_target_label.config(text=f"Current Target: {current_target:.2f} cars")
    
    def run_simulation(self):
        while self.running:
            if not self.paused:
                try:
                    self.simulation_step()
                    self.root.after(0, self.update_status, f"Running: {self.current_step} min")
                except Exception as e:
                    print(f"Simulation error: {e}")
                    self.running = False
                    break
            time.sleep(0.05)  # Control simulation speed (50ms per step)
        
        self.root.after(0, self.update_status, "Simulation Stopped")
        self.root.after(0, self.enable_start_button)
    
    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")
    
    def start_simulation(self):
        if not self.running:
            self.running = True
            self.paused = False
            
            # Only initialize if starting fresh (not after stop)
            if self.current_step == 0:
                self.initialize_simulation()
            
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.DISABLED)
            
            self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.sim_thread.start()
    
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
        self.status_label.config(text=f"Status: {'Paused' if self.paused else 'Running'}")
    
    def stop_simulation(self):
        self.running = False
        self.paused = False
        self.pause_btn.config(text="Pause")
    
    def enable_start_button(self):
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)
    
    def reset_simulation(self):
        self.running = False
        self.paused = False
        self.initialize_simulation()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.canvas.draw()
        self.status_label.config(text="Status: Ready")
        self.time_label.config(text="Time: 0 min")
        self.current_queue_label.config(text="Current Queue: --")
        self.avg_queue_label.config(text="Avg Queue (60min): --")
        self.max_queue_label.config(text="Max Queue (60min): --")
        self.current_target_label.config(text="Current Target: --")
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        self.stop_btn.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = MPCSimulator(root)
    root.mainloop()
