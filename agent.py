# agent.py
import random
import csv
import os
from datetime import datetime

def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def homeostasis(x, target=0.3, k=0.05):
    """Pull x toward target by fraction k each step."""
    return x + k * (target - x)

def temp_stress(celsius, comfort=22.0, scale=20.0):
    """|T - comfort| mapped roughly to [0,1]."""
    return clamp01(abs(celsius - comfort) / scale)

class Agent:
    def __init__(self, name="DefaultAgent"):
        self.name = name
        self.internal_state = {
            "pain": 0.2,
            "instability": 0.4,
            "need_for_control": 0.5,
            "cognitive_load": 0.4,
            "neurochem_balance": 0.6,  # higher is better
            "fatigue": 0.3,
        }
        self.environment = {
            "temperature": 22.0,
            "confinement": 0.2,
            "social_contact": 0.5,
            "noise_level": 0.3,
            "light_level": 0.6,
        }
        self.regulation = {
            "breathing": 0.3,
            "cognitive_override": 0.3,
            "pharmacology": 0.0,
            "meditation": 0.2,
            "exercise": 0.1,
        }
        self.nutrition = {
            "glucose_level": 0.7,
            "tryptophan": 0.5,
            "tyrosine": 0.5,
            "hydration": 0.8,
            "vitamin_b12": 0.7,
        }

        self.history = []   # list of dicts, one per step
        self.step = 0       # <— add a step counter

        self.params = {
            "noise_std": 0.01,
            "weights": {
                "cognitive_load": {"env": 0.35, "int": 0.30, "reg": 0.30, "nut": 0.20, "decay": 0.04},
                "pain":           {"env": 0.25, "int": 0.30, "reg": 0.20, "nut": 0.10, "decay": 0.03},
                "fatigue":        {"env": 0.15, "int": 0.25, "reg": 0.15, "nut": 0.25, "decay": 0.03},
                "neurochem_balance":{"env":0.15,"int":0.20,"reg":0.20,"nut":0.40,"decay":0.02},
                "instability":    {"env": 0.20, "int": 0.30, "reg": 0.20, "nut": 0.15, "decay": 0.02},
                "need_for_control":{"env":0.20,"int":0.25,"reg":0.25,"nut":0.10,"decay":0.03},
            }
        }

    def __str__(self):
        return f"<Agent {self.name}>"

    def _env_stress(self):
        e = self.environment
        social_deficit = 1.0 - e["social_contact"]
        return clamp01(
            0.35 * temp_stress(e["temperature"]) +
            0.25 * e["confinement"] +
            0.20 * e["noise_level"] +
            0.20 * e["light_level"] +
            0.20 * social_deficit
        )

    def _reg_relief(self):
        r = self.regulation
        return clamp01(
            0.35 * r["breathing"] +
            0.30 * r["meditation"] +
            0.25 * r["cognitive_override"] +
            0.15 * r["exercise"] +
            0.40 * r["pharmacology"]
        )

    def _nutrition_support(self):
        n = self.nutrition
        return clamp01(
            0.30 * n["glucose_level"] +
            0.30 * n["hydration"] +
            0.25 * n["tryptophan"] +
            0.20 * n["tyrosine"] +
            0.15 * n["vitamin_b12"]
        )

    def _add_noise(self):
        return random.gauss(0.0, self.params["noise_std"])

    def simulate_step(self, dt=1.0):
        """Advance the internal state by one time step and log a snapshot."""
        w = self.params["weights"]
        s = self.internal_state

        env = self._env_stress()
        reg = self._reg_relief()
        nut = self._nutrition_support()

        # helpers take k explicitly
        def pull(x, k, target=0.3):
            return homeostasis(x, target=target, k=k * dt)

        def pull_bal(x, k, target=0.6):
            return homeostasis(x, target=target, k=k * dt)

        # 1) Cognitive Load
        w_key = w["cognitive_load"]
        load_push = (w_key["env"] * env +
                     w_key["int"] * (0.6 * s["pain"] + 0.4 * s["instability"]) -
                     w_key["reg"] * reg -
                     w_key["nut"] * nut)
        s["cognitive_load"] = clamp01(pull(s["cognitive_load"], k=w_key["decay"]) + dt * load_push + self._add_noise())

        # 2) Pain
        w_key = w["pain"]
        pain_push = (w_key["env"] * (0.5 * env + 0.5 * temp_stress(self.environment["temperature"])) +
                     w_key["int"] * (0.6 * s["instability"] + 0.4 * s["fatigue"]) -
                     w_key["reg"] * (0.4 * reg + 0.4 * self.regulation["pharmacology"]) -
                     w_key["nut"] * (0.6 * self.nutrition["hydration"] + 0.4 * self.nutrition["glucose_level"]))
        s["pain"] = clamp01(pull(s["pain"], k=w_key["decay"]) + dt * pain_push + self._add_noise())

        # 3) Fatigue
        w_key = w["fatigue"]
        fatigue_push = (w_key["env"] * (0.3 * env + 0.2 * self.environment["light_level"]) +
                        w_key["int"] * (0.6 * s["cognitive_load"] + 0.4 * s["pain"]) +
                        w_key["reg"] * (0.3 * self.regulation["exercise"]) * 0.5
                        - w_key["nut"] * (0.6 * self.nutrition["glucose_level"] + 0.4 * self.nutrition["hydration"]))
        s["fatigue"] = clamp01(pull(s["fatigue"], k=w_key["decay"]) + dt * fatigue_push + self._add_noise())

        # 4) Neurochemical Balance
        w_key = w["neurochem_balance"]
        ncb_push = (- w_key["env"] * env
                    - w_key["int"] * (0.5 * s["cognitive_load"] + 0.5 * s["pain"])
                    + w_key["reg"] * (0.4 * self.regulation["meditation"] + 0.3 * self.regulation["exercise"])
                    + w_key["nut"] * (0.5 * self.nutrition["tryptophan"] + 0.4 * self.nutrition["tyrosine"] + 0.3 * self.nutrition["vitamin_b12"]))
        s["neurochem_balance"] = clamp01(pull_bal(s["neurochem_balance"], k=w_key["decay"]) + dt * ncb_push + self._add_noise())

        # 5) Instability
        w_key = w["instability"]
        inst_push = (w_key["env"] * (env + 0.2 * self.environment["noise_level"]) +
                     w_key["int"] * (0.4 * s["cognitive_load"] + 0.4 * s["fatigue"] + 0.2 * s["pain"]) -
                     w_key["reg"] * reg -
                     w_key["nut"] * nut)
        s["instability"] = clamp01(pull(s["instability"], k=w_key["decay"]) + dt * inst_push + self._add_noise())

        # 6) Need for Control
        w_key = w["need_for_control"]
        unpredictability = clamp01(0.6 * self.environment["noise_level"] + 0.4 * (1.0 - self.environment["social_contact"]))
        nfc_push = (w_key["env"] * unpredictability +
                    w_key["int"] * (0.6 * s["instability"] + 0.4 * s["pain"]) -
                    w_key["reg"] * (0.5 * self.regulation["breathing"] + 0.5 * self.regulation["meditation"]) -
                    w_key["nut"] * (0.3 * nut))
        s["need_for_control"] = clamp01(pull(s["need_for_control"], k=w_key["decay"]) + dt * nfc_push + self._add_noise())

        # increment step and log a rich snapshot
        self.step += 1
        self.history.append({
            "step": self.step,
            # internal state
            "pain": s["pain"],
            "instability": s["instability"],
            "need_for_control": s["need_for_control"],
            "cognitive_load": s["cognitive_load"],
            "neurochem_balance": s["neurochem_balance"],
            "fatigue": s["fatigue"],
            # composites
            "env_stress": env,
            "reg_relief": reg,
            "nut_support": nut,
            # (optional) raw environment/reg/nutrition at this step for traceability:
            "temperature": self.environment["temperature"],
            "confinement": self.environment["confinement"],
            "social_contact": self.environment["social_contact"],
            "noise_level": self.environment["noise_level"],
            "light_level": self.environment["light_level"],
            "breathing": self.regulation["breathing"],
            "cognitive_override": self.regulation["cognitive_override"],
            "pharmacology": self.regulation["pharmacology"],
            "meditation": self.regulation["meditation"],
            "exercise": self.regulation["exercise"],
            "glucose_level": self.nutrition["glucose_level"],
            "tryptophan": self.nutrition["tryptophan"],
            "tyrosine": self.nutrition["tyrosine"],
            "hydration": self.nutrition["hydration"],
            "vitamin_b12": self.nutrition["vitamin_b12"],
        })

    # NEW: save history to CSV
    def save_history_csv(self, filepath):
        if not self.history:
            print("No history to save.")
            return

        # Ensure folder exists
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        # Determine column order from first snapshot
        fieldnames = list(self.history[0].keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

        print(f"Saved {len(self.history)} rows to {filepath}")

import matplotlib.pyplot as plt

def plot_history(agent, show_composites=False):
    """Plot the simulation history for an agent."""
    if not agent.history:
        print("No history to plot.")
        return

    # Convert history to lists
    steps = [row["step"] for row in agent.history]

    # Internal states to plot
    internal_vars = ["pain", "instability", "need_for_control",
                     "cognitive_load", "neurochem_balance", "fatigue"]

    plt.figure(figsize=(10, 6))
    for var in internal_vars:
        plt.plot(steps, [row[var] for row in agent.history], label=var)

    if show_composites:
        composites = ["env_stress", "reg_relief", "nut_support"]
        for var in composites:
            plt.plot(steps, [row[var] for row in agent.history],
                     linestyle="--", label=var)

    plt.ylim(0, 1)
    plt.xlabel("Step")
    plt.ylabel("Value (0–1)")
    plt.title(f"Agent Simulation: {agent.name}")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    agent = Agent(name="Athena")
    print(agent)

    for _ in range(200):
        agent.simulate_step(dt=1.0)

    print("Final Internal State:")
    for k, v in agent.internal_state.items():
        print(f"  {k:18s} {v:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("data", f"{agent.name.lower()}_run_{timestamp}.csv")
    agent.save_history_csv(out_path)

    # NEW: plot
    plot_history(agent, show_composites=True)

    input("\nPress Enter to exit...")
