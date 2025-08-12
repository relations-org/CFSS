# agent.py
import random, csv, os, json, argparse, math
from datetime import datetime

# ---------- defaults & helpers ----------
DEFAULT_CONFIG = {
    "run": {
        "steps": 200,
        "agent_name": "Athena",
        "data_dir": "data",
        "log_csv": True
    },
    "internal_state": {
        "pain": 0.2,
        "instability": 0.4,
        "need_for_control": 0.5,
        "cognitive_load": 0.4,
        "neurochem_balance": 0.6,
        "fatigue": 0.3
    },
    "environment": {
        "temperature": 22.0,  # °C
        "confinement": 0.2,   # 0..1
        "social_contact": 0.5,# 0..1
        "noise_level": 0.3,   # 0..1
        "light_level": 0.6    # 0..1 (blue/bright)
    },
    "regulation": {
        "breathing": 0.3,
        "cognitive_override": 0.3,
        "pharmacology": 0.0,
        "meditation": 0.2,
        "exercise": 0.1
    },
    "nutrition": {
        "glucose_level": 0.7,
        "tryptophan": 0.5,
        "tyrosine": 0.5,
        "hydration": 0.8,
        "vitamin_b12": 0.7
    },
    "params": {
        "noise_std": 0.01,
        # variable-specific homeostasis targets (0..1)
        "targets": {
            "pain": 0.20,
            "instability": 0.20,
            "need_for_control": 0.35,
            "cognitive_load": 0.25,
            "neurochem_balance": 0.70,  # "good" variable
            "fatigue": 0.25
        },
        # environment update knobs
        "env_update": {
            "comfort_temperature": 22.0,
            "env_step": 0.05,            # how strongly actions move env vars per step
            "dispersion_noise": 0.02     # scales randomness in env changes
        },
        # weights for internal-variable updates
        "weights": {
            "cognitive_load": {"env": 0.35, "int": 0.30, "reg": 0.30, "nut": 0.20, "decay": 0.05},
            "pain":           {"env": 0.25, "int": 0.30, "reg": 0.20, "nut": 0.10, "decay": 0.05},
            "fatigue":        {"env": 0.15, "int": 0.25, "reg": 0.15, "nut": 0.25, "decay": 0.05},
            "neurochem_balance":{"env":0.15,"int":0.20,"reg":0.20,"nut":0.40,"decay": 0.04},
            "instability":    {"env": 0.20, "int": 0.30, "reg": 0.20, "nut": 0.15, "decay": 0.04},
            "need_for_control":{"env":0.20,"int":0.25,"reg":0.25,"nut":0.10,"decay": 0.05}
        }
    }
}

def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def homeostasis(x, target=0.3, k=0.05):
    """Pull x toward target by fraction k each step."""
    return x + k * (target - x)

def temp_stress(celsius, comfort=22.0, scale=20.0):
    """|T - comfort| mapped roughly to [0,1]."""
    return clamp01(abs(celsius - comfort) / scale)

def deep_merge(base, add):
    if not isinstance(base, dict) or not isinstance(add, dict):
        return add
    out = dict(base)
    for k, v in add.items():
        out[k] = deep_merge(base.get(k), v) if k in base else v
    return out

def load_config(path):
    if not path or not os.path.exists(path):
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    return deep_merge(DEFAULT_CONFIG, user_cfg)

# soft factors for stability
def sat_factor(x, floor=0.1):
    """Diminishing returns for pushes near edges: peaks at x=0.5, min near edges."""
    return floor + (1.0 - floor) * (4.0 * x * (1.0 - x))  # in [floor,1]

def edge_factor(x):
    """High near edges, low in the middle; used to boost noise a bit near walls."""
    return clamp01(1.0 - 4.0 * x * (1.0 - x))  # 1 at edges, 0 at center

def inv_u(x):
    """Inverted-U (0..1), peak at 0.5."""
    return 4.0 * x * (1.0 - x)

# ---------- Agent ----------
class Agent:
    def __init__(self, config):
        run = config.get("run", {})
        self.name = run.get("agent_name", "Agent")
        self.internal_state = dict(config["internal_state"])
        self.environment   = dict(config["environment"])
        self.regulation    = dict(config["regulation"])
        self.nutrition     = dict(config["nutrition"])
        self.params        = dict(config["params"])
        self.targets       = dict(self.params.get("targets", {}))
        self.envp          = dict(self.params.get("env_update", {}))

        self.history = []
        self.step = 0

    def __str__(self): return f"<Agent {self.name}>"

    # --- composites ---
    def _env_stress(self):
        e = self.environment
        social_deficit = 1.0 - e["social_contact"]
        return clamp01(
            0.35 * temp_stress(e["temperature"], self.envp.get("comfort_temperature", 22.0)) +
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

    def _noise_for(self, x):
        std = self.params["noise_std"] * (0.5 + 0.5 * edge_factor(x))
        return random.gauss(0.0, std)

    # --- agent acts on environment ---
    def _motivation_ability_dispersion(self, env_stress, reg_relief, nut_support):
        s = self.internal_state
        # motive: pain(+), need_for_control(inverted-U +), cognitive_load(+),
        #         env_stress(+), reg_relief(-), fatigue(-), neurochem_balance(-)
        m = (
            0.25 * s["pain"]
          + 0.20 * inv_u(s["need_for_control"])
          + 0.15 * s["cognitive_load"]
          + 0.20 * env_stress
          - 0.10 * reg_relief
          - 0.10 * s["fatigue"]
          - 0.10 * s["neurochem_balance"]
        )
        m = clamp01(m)  # 0..1

        # ability: neurochem_balance(+), nut_support(+), reg_relief(+),
        #          fatigue(-), pain(-), cognitive_load(-), env_stress(-),
        #          need_for_control (inverted-U + small)
        a = (
            0.30 * s["neurochem_balance"]
          + 0.25 * nut_support
          + 0.20 * reg_relief
          - 0.20 * s["fatigue"]
          - 0.15 * s["pain"]
          - 0.10 * s["cognitive_load"]
          - 0.10 * env_stress
          + 0.10 * inv_u(s["need_for_control"])
        )
        a = clamp01(a)

        # dispersion (range of action variability): up with instability & env_stress,
        # down with reg_relief; small boost from need_for_control
        disp = (
            0.50
          + 0.30 * s["instability"]
          + 0.20 * env_stress
          - 0.20 * reg_relief
          + 0.10 * s["need_for_control"]
        )
        disp = max(0.3, min(1.7, disp))  # keep sane

        return m, a, disp

    def _update_environment(self, motive, ability, dispersion, dt):
        """Agent nudges environment toward lower env_stress."""
        act = motive * ability  # 0..1
        if act <= 0.0:
            return

        step = self.envp.get("env_step", 0.05) * act * dt
        comfort_T = self.envp.get("comfort_temperature", 22.0)
        disp_noise = self.envp.get("dispersion_noise", 0.02) * dispersion

        e = self.environment

        # Move temperature toward comfort
        dT = step * (comfort_T - e["temperature"]) + random.gauss(0.0, disp_noise * 2.0)
        e["temperature"] += dT  # not clamped (open-range), but drifts toward comfort

        # Reduce confinement, noise, and blue/bright light; increase social_contact
        def nudgedown(x): 
            dx = -step * (0.25 + 0.75 * x) + random.gauss(0.0, disp_noise)
            return clamp01(x + dx)

        def nudgeup(x): 
            dx = step * (0.25 + 0.75 * (1.0 - x)) + random.gauss(0.0, disp_noise)
            return clamp01(x + dx)

        e["confinement"]   = nudgedown(e["confinement"])
        e["noise_level"]   = nudgedown(e["noise_level"])
        e["light_level"]   = nudgedown(e["light_level"])
        e["social_contact"]= nudgeup(e["social_contact"])

    # --- simulate one step ---
    def simulate_step(self, dt=1.0):
        w = self.params["weights"]
        s = self.internal_state

        env = self._env_stress()
        reg = self._reg_relief()
        nut = self._nutrition_support()

        # helpers with per-variable targets
        def pull(x, k, var):
            target = self.targets.get(var, 0.3)
            return homeostasis(x, target=target, k=k * dt)

        # 1) Cognitive Load (bad -> push scales down near edges)
        w_key = w["cognitive_load"]
        load_push = (w_key["env"] * env +
                     w_key["int"] * (0.6 * s["pain"] + 0.4 * s["instability"]) -
                     w_key["reg"] * reg -
                     w_key["nut"] * nut)
        load_push *= sat_factor(s["cognitive_load"])
        s["cognitive_load"] = clamp01(
            pull(s["cognitive_load"], w_key["decay"], "cognitive_load")
            + dt * load_push
            + self._noise_for(s["cognitive_load"])
        )

        # 2) Pain (bad)
        w_key = w["pain"]
        pain_push = (w_key["env"] * (0.5 * env + 0.5 * temp_stress(self.environment["temperature"], self.envp.get("comfort_temperature",22.0))) +
                     w_key["int"] * (0.6 * s["instability"] + 0.4 * s["fatigue"]) -
                     w_key["reg"] * (0.4 * reg + 0.4 * self.regulation["pharmacology"]) -
                     w_key["nut"] * (0.6 * self.nutrition["hydration"] + 0.4 * self.nutrition["glucose_level"]))
        pain_push *= sat_factor(s["pain"])
        s["pain"] = clamp01(
            pull(s["pain"], w_key["decay"], "pain")
            + dt * pain_push
            + self._noise_for(s["pain"])
        )

        # 3) Fatigue (bad)
        w_key = w["fatigue"]
        fatigue_push = (w_key["env"] * (0.3 * env + 0.2 * self.environment["light_level"]) +
                        w_key["int"] * (0.6 * s["cognitive_load"] + 0.4 * s["pain"]) +
                        w_key["reg"] * (0.3 * self.regulation["exercise"]) * 0.5
                        - w_key["nut"] * (0.6 * self.nutrition["glucose_level"] + 0.4 * self.nutrition["hydration"]))
        fatigue_push *= sat_factor(s["fatigue"])
        s["fatigue"] = clamp01(
            pull(s["fatigue"], w_key["decay"], "fatigue")
            + dt * fatigue_push
            + self._noise_for(s["fatigue"])
        )

        # 4) Neurochem Balance (good)
        w_key = w["neurochem_balance"]
        ncb_push = (- w_key["env"] * env
                    - w_key["int"] * (0.5 * s["cognitive_load"] + 0.5 * s["pain"])
                    + w_key["reg"] * (0.4 * self.regulation["meditation"] + 0.3 * self.regulation["exercise"])
                    + w_key["nut"] * (0.5 * self.nutrition["tryptophan"] + 0.4 * self.nutrition["tyrosine"] + 0.3 * self.nutrition["vitamin_b12"]))
        ncb_push *= sat_factor(s["neurochem_balance"])
        s["neurochem_balance"] = clamp01(
            pull(s["neurochem_balance"], w_key["decay"], "neurochem_balance")
            + dt * ncb_push
            + self._noise_for(s["neurochem_balance"])
        )

        # 5) Instability (bad)
        w_key = w["instability"]
        inst_push = (w_key["env"] * (env + 0.2 * self.environment["noise_level"]) +
                     w_key["int"] * (0.4 * s["cognitive_load"] + 0.4 * s["fatigue"] + 0.2 * s["pain"]) -
                     w_key["reg"] * reg -
                     w_key["nut"] * nut)
        inst_push *= sat_factor(s["instability"])
        s["instability"] = clamp01(
            pull(s["instability"], w_key["decay"], "instability")
            + dt * inst_push
            + self._noise_for(s["instability"])
        )

        # 6) Need for Control (bad-ish; inverted-U influences handled in env action)
        w_key = w["need_for_control"]
        unpredictability = clamp01(0.6 * self.environment["noise_level"] + 0.4 * (1.0 - self.environment["social_contact"]))
        nfc_push = (w_key["env"] * unpredictability +
                    w_key["int"] * (0.6 * s["instability"] + 0.4 * s["pain"]) -
                    w_key["reg"] * (0.5 * self.regulation["breathing"] + 0.5 * self.regulation["meditation"]) -
                    w_key["nut"] * (0.3 * nut))
        nfc_push *= sat_factor(s["need_for_control"])
        s["need_for_control"] = clamp01(
            pull(s["need_for_control"], w_key["decay"], "need_for_control")
            + dt * nfc_push
            + self._noise_for(s["need_for_control"])
        )

        # --- agent modifies environment (CLOSE THE LOOP) ---
        motive, ability, disp = self._motivation_ability_dispersion(env, reg, nut)
        self._update_environment(motive, ability, disp, dt)

        # recompute composites AFTER environment move (for logging)
        env2 = self._env_stress()
        reg2 = self._reg_relief()
        nut2 = self._nutrition_support()

        # log snapshot
        self.step += 1
        self.history.append({
            "step": self.step,
            "pain": s["pain"],
            "instability": s["instability"],
            "need_for_control": s["need_for_control"],
            "cognitive_load": s["cognitive_load"],
            "neurochem_balance": s["neurochem_balance"],
            "fatigue": s["fatigue"],
            # composites (post-env-update)
            "env_stress": env2,
            "reg_relief": reg2,
            "nut_support": nut2,
            # raw env/reg/nut
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
            # action diagnostics
            "motive": motive,
            "ability": ability,
            "dispersion": disp
        })

    def save_history_csv(self, filepath):
        if not self.history:
            print("No history to save.")
            return
        folder = os.path.dirname(filepath)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        fieldnames = list(self.history[0].keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        print(f"Saved {len(self.history)} rows to {filepath}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="CFSS agent simulator")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config")
    p.add_argument("--steps", type=int, default=None, help="Override steps")
    p.add_argument("--name", type=str, default=None, help="Override agent name")
    p.add_argument("--outdir", type=str, default=None, help="Override output dir")
    p.add_argument("--no-pause", action="store_true", help="Do not wait for Enter at the end")
    p.add_argument("--plot", action="store_true", help="Show a matplotlib plot after run")
    return p.parse_args()

def maybe_plot(agent, show_composites=True):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}")
        return
    if not agent.history:
        print("[plot] no history to plot")
        return
    steps = [row["step"] for row in agent.history]
    internal_vars = ["pain","instability","need_for_control","cognitive_load","neurochem_balance","fatigue"]
    plt.figure(figsize=(10,6))
    for var in internal_vars:
        plt.plot(steps, [row[var] for row in agent.history], label=var)
    if show_composites:
        for var in ["env_stress","reg_relief","nut_support"]:
            plt.plot(steps, [row[var] for row in agent.history], linestyle="--", label=var)
    plt.ylim(0,1); plt.xlabel("Step"); plt.ylabel("Value (0–1)")
    plt.title(f"Agent Simulation: {agent.name}")
    plt.legend(fontsize="small"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides for GUI convenience
    if args.steps is not None:
        cfg.setdefault("run", {})["steps"] = args.steps
    if args.name is not None:
        cfg.setdefault("run", {})["agent_name"] = args.name
    if args.outdir is not None:
        cfg.setdefault("run", {})["data_dir"] = args.outdir

    agent = Agent(cfg)
    print(agent)

    steps = cfg["run"]["steps"]
    for _ in range(steps):
        agent.simulate_step(dt=1.0)

    print("Final Internal State:")
    for k, v in agent.internal_state.items():
        print(f"  {k:18s} {v:.3f}")

    if cfg["run"].get("log_csv", True):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = cfg["run"]["data_dir"]
        out_path = os.path.join(out_dir, f"{agent.name.lower()}_run_{ts}.csv")
        agent.save_history_csv(out_path)

    if args.plot:
        maybe_plot(agent, show_composites=True)

    if not args.no_pause:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
