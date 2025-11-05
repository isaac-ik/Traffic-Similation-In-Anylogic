import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# --- Predictive Control Model ---
class TrafficMPCSim:
    def __init__(self):
        self.dt = 1.0
        self.lambda_max = 5.0
        self.mu = 6.0
        self.q_ref = 20.0
        self.N_horizon = 10
        self.time_steps = 120
        self.Q = np.array([[1.0]])
        self.R = np.array([[0.5]])
        self.A = np.array([[1.0]])
        self.B = np.array([[self.dt]])

        # Compute LQR gain
        self.P = self.Q.copy()
        for _ in range(self.N_horizon):
            K = np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ (self.B.T @ self.P @ self.A)
            self.P = self.Q + self.A.T @ self.P @ (self.A - self.B @ K)
        self.K = K

        # Data buffers
        self.reset()

    def reset(self):
        self.t = 0
        self.q = [10.0]
        self.u1 = [0.5]
        self.u2 = [0.5]
        self.inflow = [0.0]
        self.outflow = [0.0]

    def step(self):
        q_prev = self.q[-1]
        q_error = np.array([[q_prev - self.q_ref]])  # Make q_error a 2D array
        u_raw = -self.K @ q_error
        u1 = np.clip(1 - 0.2*u_raw[0,0], 0, 1)  # Extract scalar from array
        u2 = np.clip(1 + 0.2*u_raw[0,0], 0, 1)  # Extract scalar from array

        w = np.random.randn() * 0.3
        inflow = self.lambda_max * u1
        outflow = self.mu * u2
        q_new = max(0, q_prev + self.dt * (inflow - outflow) + w)

        self.q.append(float(q_new))
        self.u1.append(float(u1))
        self.u2.append(float(u2))
        self.inflow.append(float(inflow))
        self.outflow.append(float(outflow))
        self.t += 1

# --- GUI ---
class TrafficMPCGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictive Traffic Control System")
        self.sim = TrafficMPCSim()
        self.running = False

        # Matplotlib figure
        self.fig = Figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Controls frame
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(control_frame, text="Target Queue (cars):").pack(side=tk.LEFT, padx=5)
        self.q_ref_slider = ttk.Scale(control_frame, from_=5, to=50, orient="horizontal", command=self.update_target)
        self.q_ref_slider.set(self.sim.q_ref)
        self.q_ref_slider.pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_sim)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_sim)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        self.reset_btn = ttk.Button(control_frame, text="Reset", command=self.reset_sim)
        self.reset_btn.pack(side=tk.LEFT, padx=10)

    def update_target(self, val):
        self.sim.q_ref = float(val)

    def start_sim(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.run_sim, daemon=True).start()

    def stop_sim(self):
        self.running = False

    def reset_sim(self):
        self.running = False
        self.sim.reset()
        self.update_plot()

    def run_sim(self):
        while self.running and self.sim.t < self.sim.time_steps:
            self.sim.step()
            self.update_plot()
            time.sleep(0.2)  # reduced delay for better responsiveness

    def update_plot(self):
        t_vals = range(len(self.sim.q))
        self.ax1.cla(); self.ax2.cla(); self.ax3.cla()

        # Queue vs Target
        self.ax1.plot(t_vals, self.sim.q, label="Queue", color="blue")
        self.ax1.axhline(self.sim.q_ref, color="red", linestyle="--", label="Target")
        self.ax1.set_title("Queue Length vs Target")
        self.ax1.set_ylabel("Queue (cars)")
        self.ax1.legend()
        self.ax1.grid(True)

        # Control inputs
        self.ax2.plot(t_vals, self.sim.u1, label="Metering (u1)")
        self.ax2.plot(t_vals, self.sim.u2, label="Green Fraction (u2)")
        self.ax2.set_title("Control Inputs")
        self.ax2.set_ylabel("Control Value")
        self.ax2.legend()
        self.ax2.grid(True)

        # Flows
        self.ax3.plot(t_vals, self.sim.inflow, label="Inflow")
        self.ax3.plot(t_vals, self.sim.outflow, label="Outflow")
        self.ax3.set_title("Traffic Flows")
        self.ax3.set_xlabel("Time (min)")
        self.ax3.set_ylabel("Cars/min")
        self.ax3.legend()
        self.ax3.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficMPCGUI(root)
    root.mainloop()
