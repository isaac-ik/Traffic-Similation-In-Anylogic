import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

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
        
        # Default parameters
        self.dt = 1.0
        self.lambda_nominal = 10.0
        self.mu = 15.0
        self.N = 10
        self.sim_minutes = 120
        self.q0 = 20.0
        self.q_ref_1 = 5.0
        self.q_ref_2 = 8.0
        self.q_ref_switch = 60
        
        # Setup UI
        self.setup_ui()
        self.initialize_simulation()
        
    def setup_ui(self):
        # Control panel (left side)
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Parameters
        ttk.Label(control_frame, text="Simulation Parameters", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=5)
        
        params = [
            ("Initial Queue (q0):", "q0", 0, 50, 20.0),
            ("Target Queue 1:", "q_ref_1", 0, 20, 5.0),
            ("Target Queue 2:", "q_ref_2", 0, 20, 8.0),
            ("Switch Time (min):", "q_ref_switch", 0, 120, 60),
            ("Inflow Rate (λ):", "lambda_nominal", 1, 20, 10.0),
            ("Outflow Rate (μ):", "mu", 1, 30, 15.0),
            ("Horizon (N):", "N", 1, 30, 10),
            ("Sim Duration (min):", "sim_minutes", 10, 300, 120)
        ]
        
        self.sliders = {}
        self.labels = {}
        
        for i, (label, var, min_val, max_val, default) in enumerate(params):
            row = i + 1
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
        button_frame.grid(row=len(params)+1, column=0, columnspan=3, pady=20)
        
        self.start_btn = ttk.Button(button_frame, text="Start", command=self.start_simulation)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.pause_btn = ttk.Button(button_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.grid(row=0, column=1, padx=5)
        
        self.reset_btn = ttk.Button(button_frame, text="Reset", command=self.reset_simulation)
        self.reset_btn.grid(row=0, column=2, padx=5)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Ready", font=('Arial', 10))
        self.status_label.grid(row=len(params)+2, column=0, columnspan=3, pady=10)
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="10")
        stats_frame.grid(row=len(params)+3, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        self.avg_queue_label = ttk.Label(stats_frame, text="Avg Queue: --")
        self.avg_queue_label.grid(row=0, column=0, sticky=tk.W)
        
        self.max_queue_label = ttk.Label(stats_frame, text="Max Queue: --")
        self.max_queue_label.grid(row=1, column=0, sticky=tk.W)
        
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
        val = float(value)
        if var in ['N', 'q_ref_switch', 'sim_minutes']:
            val = int(val)
        setattr(self, var, val)
        label.config(text=f"{val:.1f}" if isinstance(val, float) else f"{val}")
        
    def initialize_simulation(self):
        self.A = np.array([[1.0]])
        self.B = np.array([[self.dt * self.lambda_nominal, -self.dt * self.mu]])
        self.Q = np.array([[1.0]])
        self.R = np.diag([0.1, 0.1])
        
        self.Qhist = np.zeros(self.sim_minutes + 1)
        self.Uhist = np.zeros((self.sim_minutes, 2))
        self.q_ref = np.zeros(self.sim_minutes + 1)
        self.q_ref[:self.q_ref_switch] = self.q_ref_1
        self.q_ref[self.q_ref_switch:] = self.q_ref_2
        
        self.Qhist[0] = self.q0
        self.current_step = 0
        
    def finite_horizon_lqr_gains(self):
        P = [None] * (self.N + 1)
        K = [None] * self.N
        P[self.N] = self.Q.copy()
        
        for k in range(self.N-1, -1, -1):
            S = self.R + self.B.T @ P[k+1] @ self.B
            Kk = np.linalg.solve(S, self.B.T @ P[k+1] @ self.A)
            K[k] = Kk
            P[k] = self.Q + self.A.T @ P[k+1] @ (self.A - self.B @ Kk)
        
        return K, P
    
    def simulation_step(self):
        if self.current_step >= self.sim_minutes:
            self.stop_simulation()
            return
        
        t = self.current_step
        K_seq, _ = self.finite_horizon_lqr_gains()
        
        q = self.Qhist[t]
        q_err = q - self.q_ref[t]
        K0 = K_seq[0]
        u = -K0 @ np.array([[q_err]])
        u = u.ravel()
        u = np.clip(u, 0.0, 1.0)
        
        self.Uhist[t] = u
        
        # Simulate dynamics with noise
        w = np.random.normal(scale=0.2)
        q_next = (self.A @ np.array([[q]]) + (self.B @ u.reshape(2,1))).item() + w
        q_next = max(q_next, 0.0)
        
        self.Qhist[t+1] = q_next
        self.current_step += 1
        
        self.update_plots()
        
    def update_plots(self):
        time = np.arange(self.current_step + 1)
        
        self.ax1.clear()
        self.ax1.plot(time, self.Qhist[:self.current_step+1], 'b-', linewidth=2, label='Queue')
        self.ax1.plot(time, self.q_ref[:self.current_step+1], 'r--', label='Target')
        self.ax1.set_ylabel('Queued Cars')
        self.ax1.set_xlabel('Time (minutes)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Queue Evolution')
        
        if self.current_step > 0:
            time_ctrl = np.arange(self.current_step)
            self.ax2.clear()
            self.ax2.step(time_ctrl, self.Uhist[:self.current_step, 0], where='post', label='Metering (u1)', linewidth=2)
            self.ax2.step(time_ctrl, self.Uhist[:self.current_step, 1], where='post', label='Green (u2)', linewidth=2)
            self.ax2.set_ylabel('Control (fraction)')
            self.ax2.set_xlabel('Time (minutes)')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(-0.1, 1.1)
            self.ax2.set_title('Control Inputs')
            
            inflow = self.Uhist[:self.current_step, 0] * self.lambda_nominal
            outflow = self.Uhist[:self.current_step, 1] * self.mu
            
            self.ax3.clear()
            self.ax3.step(time_ctrl, inflow, where='post', label='Inflow', linewidth=2)
            self.ax3.step(time_ctrl, outflow, where='post', label='Outflow', linewidth=2)
            self.ax3.set_ylabel('Cars per minute')
            self.ax3.set_xlabel('Time (minutes)')
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)
            self.ax3.set_title('Flow Rates')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update statistics
        if self.current_step > 0:
            avg_q = np.mean(self.Qhist[:self.current_step+1])
            max_q = np.max(self.Qhist[:self.current_step+1])
            self.avg_queue_label.config(text=f"Avg Queue: {avg_q:.2f} cars")
            self.max_queue_label.config(text=f"Max Queue: {max_q:.2f} cars")
        
    def run_simulation(self):
        while self.running and self.current_step < self.sim_minutes:
            if not self.paused:
                self.simulation_step()
                self.root.after(0, self.update_status, f"Running: {self.current_step}/{self.sim_minutes}")
            self.root.after(50)  # Control simulation speed
        
        if self.current_step >= self.sim_minutes:
            self.root.after(0, self.update_status, "Simulation Complete")
            self.running = False
            self.root.after(0, self.enable_buttons)
    
    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")
    
    def start_simulation(self):
        if not self.running:
            self.running = True
            self.paused = False
            self.initialize_simulation()
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.DISABLED)
            
            # Disable sliders
            for slider in self.sliders.values():
                slider.config(state=tk.DISABLED)
            
            self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
            self.sim_thread.start()
    
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")
        self.status_label.config(text=f"Status: {'Paused' if self.paused else 'Running'}")
    
    def stop_simulation(self):
        self.running = False
        self.enable_buttons()
    
    def enable_buttons(self):
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        self.reset_btn.config(state=tk.NORMAL)
        for slider in self.sliders.values():
            slider.config(state=tk.NORMAL)
    
    def reset_simulation(self):
        self.running = False
        self.paused = False
        self.initialize_simulation()
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.canvas.draw()
        self.status_label.config(text="Status: Ready")
        self.avg_queue_label.config(text="Avg Queue: --")
        self.max_queue_label.config(text="Max Queue: --")
        self.enable_buttons()

if __name__ == "__main__":
    root = tk.Tk()
    app = MPCSimulator(root)
    root.mainloop()
