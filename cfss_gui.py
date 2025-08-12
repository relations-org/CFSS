# cfss_gui.py
import json, os, sys, subprocess, threading, queue, glob, csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import importlib.util

SECTIONS = ["internal_state", "environment", "regulation", "nutrition"]
WEIGHTS_KEY = ("params", "weights")
FLOAT_OPEN_RANGE_KEYS = {("environment", "temperature")}

def app_dir():
    return os.path.dirname(os.path.abspath(__file__))

def default_config_path():
    return os.path.join(app_dir(), "cfss_config.json")

def import_agent_module():
    path = os.path.join(app_dir(), "agent.py")
    spec = importlib.util.spec_from_file_location("agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def merge_defaults(base, defaults):
    if not isinstance(defaults, dict):
        return base if base is not None else defaults
    out = dict(defaults)
    if isinstance(base, dict):
        for k, v in base.items():
            if isinstance(v, dict):
                out[k] = merge_defaults(v, defaults.get(k, {}))
            else:
                out[k] = v
    return out

def is_open_range(section, key):
    return (section, key) in FLOAT_OPEN_RANGE_KEYS

def clamp01(x):
    try: xf = float(x)
    except: return 0.0
    return max(0.0, min(1.0, xf))

class CFSSGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CFSS – Config, Runner, Data")
        self.geometry("1280x800")
        self.minsize(1100, 700)

        # theme: green-on-black
        self._apply_green_theme()

        # state
        self.config_path = tk.StringVar(value=default_config_path())
        self.agent_mod = None
        self.defaults = {}
        self.cfg = {}
        self.entry_vars = {}
        self.weight_vars = {}
        self.run_steps = tk.StringVar(value="200")
        self.run_name  = tk.StringVar(value="Athena")
        self.data_dir  = tk.StringVar(value="data")
        self.log_csv   = tk.BooleanVar(value=True)

        self.proc = None
        self.output_queue = queue.Queue()

        # build UI
        self._build_ui()

        # load defaults and config
        self._load_defaults_from_agent()
        self._load_cfg(self.config_path.get())

        # populate
        self.populate_from_config()
        self.refresh_data_list()

    # ---------- theme ----------
    def _apply_green_theme(self):
        self.configure(bg="black")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except:
            pass
        # Base colors
        fg = "#00ff66"
        bg = "black"
        style.configure(".", background=bg, foreground=fg, fieldbackground=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background="#003300", foreground=fg)
        style.map("TButton",
                  foreground=[("disabled", "#226622"), ("active", "#00ff99")],
                  background=[("active", "#004400")])
        style.configure("TEntry", fieldbackground="#001100", foreground=fg)
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", background="#002200", foreground=fg)
        style.map("TNotebook.Tab", background=[("selected", "#003300")])

        # default font: Consolas if present
        consolas = ("Consolas", 10)
        self.option_add("*Font", consolas)
        self.option_add("*Text.Font", ("Consolas", 10))
        self.option_add("*Entry.Font", ("Consolas", 10))
        self.option_add("*Treeview.Font", ("Consolas", 10))
        self.option_add("*Treeview.Heading.Font", ("Consolas", 10, "bold"))

    # ---------- building UI ----------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        ttk.Label(top, text="Config:").pack(side=tk.LEFT)
        tk.Entry(top, textvariable=self.config_path, width=60, bg="#001100", fg="#00ff66", insertbackground="#00ff66").pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Browse…", command=self.choose_config_file).pack(side=tk.LEFT)
        ttk.Button(top, text="Load", command=self.load_config_btn).pack(side=tk.LEFT, padx=(10,0))
        ttk.Button(top, text="Save", command=self.save_config_btn).pack(side=tk.LEFT, padx=6)

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Tab 1: Setup/Run
        self.tab_setup = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_setup, text="Setup & Run")

        # Tab 2: Data Viewer
        self.tab_data = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_data, text="Data Viewer")

        # ----- Setup & Run layout -----
        main = ttk.Frame(self.tab_setup)
        main.pack(fill=tk.BOTH, expand=True)

        self.left_col  = ttk.LabelFrame(main, text="Initial Setup")
        self.middle_col= ttk.LabelFrame(main, text="Step Rules (Weights)")
        self.right_col = ttk.LabelFrame(main, text="Run Options")

        self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0,6), pady=(0,6))
        self.middle_col.grid(row=0, column=1, sticky="nsew", padx=6, pady=(0,6))
        self.right_col.grid(row=0, column=2, sticky="nsew", padx=(6,0), pady=(0,6))

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.columnconfigure(2, weight=1)
        main.rowconfigure(0, weight=1)

        self._build_left_col()
        self._build_middle_col()
        self._build_right_col()

        # Console
        console_frame = ttk.LabelFrame(self.tab_setup, text="Console")
        console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        self.console = tk.Text(console_frame, height=10, wrap="word", bg="black", fg="#00ff66", insertbackground="#00ff66")
        self.console.pack(fill=tk.BOTH, expand=True)

        # drain output
        self.after(100, self._drain_output_queue)

        # ----- Data Viewer layout -----
        self._build_data_viewer()

    def _build_left_col(self):
        canvas = tk.Canvas(self.left_col, highlightthickness=0, bg="black")
        vsb = ttk.Scrollbar(self.left_col, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set, bg="black")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        for section in SECTIONS:
            box = ttk.LabelFrame(frame, text=section)
            box.pack(fill=tk.X, padx=6, pady=6)
            grid_row = 0
            self.entry_vars.setdefault(section, {})
            for key in []:  # populated later
                pass
            # will be populated in populate_from_config()

        self.left_inner_frame = frame  # for later repopulation

    def _build_middle_col(self):
        header = ttk.Frame(self.middle_col)
        header.pack(fill=tk.X, padx=6, pady=(6,0))
        for i, c in enumerate(["variable","env","int","reg","nut","decay"]):
            ttk.Label(header, text=c, width=12 if i==0 else 8).grid(row=0, column=i, sticky="w", padx=4)

        canvas = tk.Canvas(self.middle_col, highlightthickness=0, bg="black")
        vsb = ttk.Scrollbar(self.middle_col, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set, bg="black")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6,0), pady=6)
        vsb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0,6), pady=6)

        self.weights_frame = frame  # populate later

    def _build_right_col(self):
        frm = ttk.Frame(self.right_col)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ttk.Label(frm, text="Agent name").grid(row=0, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.run_name, bg="#001100", fg="#00ff66", insertbackground="#00ff66").grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(frm, text="Steps").grid(row=1, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.run_steps, bg="#001100", fg="#00ff66", insertbackground="#00ff66").grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(frm, text="Data dir").grid(row=2, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.data_dir, bg="#001100", fg="#00ff66", insertbackground="#00ff66").grid(row=2, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(frm, text="Browse…", command=self.choose_data_dir).grid(row=2, column=2, padx=4)

        ttk.Checkbutton(frm, text="Log CSV", variable=self.log_csv).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6,6))

        ttk.Separator(frm).grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)

        btns = ttk.Frame(frm)
        btns.grid(row=5, column=0, columnspan=3, sticky="ew")
        self.run_btn = ttk.Button(btns, text="GO!", command=self.run_simulation)
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,6))
        ttk.Button(btns, text="Clear Console", command=lambda: self.console.delete("1.0", tk.END)).pack(side=tk.LEFT)

        frm.columnconfigure(1, weight=1)

    def _build_data_viewer(self):
        outer = ttk.Frame(self.tab_data)
        outer.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(outer)
        right = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="CSV files").pack(anchor="w")
        self.csv_list = tk.Listbox(left, width=40, height=25, bg="black", fg="#00ff66", selectbackground="#004400")
        self.csv_list.pack(fill=tk.Y, expand=False)
        self.csv_list.bind("<<ListboxSelect>>", lambda e: self.preview_selected_csv())

        ctrl = ttk.Frame(left); ctrl.pack(fill=tk.X, pady=6)
        ttk.Button(ctrl, text="Refresh", command=self.refresh_data_list).pack(side=tk.LEFT, padx=(0,6))
        ttk.Button(ctrl, text="Open", command=self.open_selected_csv).pack(side=tk.LEFT)

        # preview area (Treeview)
        self.preview = ttk.Treeview(right, columns=(), show="headings")
        self.preview.pack(fill=tk.BOTH, expand=True)
        ysb = ttk.Scrollbar(right, orient="vertical", command=self.preview.yview)
        self.preview.configure(yscroll=ysb.set)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------- defaults & config ----------
    def _load_defaults_from_agent(self):
        try:
            self.agent_mod = import_agent_module()
            self.defaults = dict(self.agent_mod.DEFAULT_CONFIG)
        except Exception as e:
            messagebox.showwarning("Defaults", f"Could not import agent.py defaults.\n{e}")
            self.defaults = {
                "run": {"steps": 200, "agent_name": "Athena", "data_dir": "data", "log_csv": True},
                "internal_state": {"pain":0.2,"instability":0.4,"need_for_control":0.5,"cognitive_load":0.4,"neurochem_balance":0.6,"fatigue":0.3},
                "environment": {"temperature":22.0,"confinement":0.2,"social_contact":0.5,"noise_level":0.3,"light_level":0.6},
                "regulation": {"breathing":0.3,"cognitive_override":0.3,"pharmacology":0.0,"meditation":0.2,"exercise":0.1},
                "nutrition": {"glucose_level":0.7,"tryptophan":0.5,"tyrosine":0.5,"hydration":0.8,"vitamin_b12":0.7},
                "params": {"noise_std":0.01,"weights":{
                    "cognitive_load":{"env":0.35,"int":0.30,"reg":0.30,"nut":0.20,"decay":0.04},
                    "pain":{"env":0.25,"int":0.30,"reg":0.20,"nut":0.10,"decay":0.03},
                    "fatigue":{"env":0.15,"int":0.25,"reg":0.15,"nut":0.25,"decay":0.03},
                    "neurochem_balance":{"env":0.15,"int":0.20,"reg":0.20,"nut":0.40,"decay":0.02},
                    "instability":{"env":0.20,"int":0.30,"reg":0.20,"nut":0.15,"decay":0.02},
                    "need_for_control":{"env":0.20,"int":0.25,"reg":0.25,"nut":0.10,"decay":0.03}
                }}
            }

    def _load_cfg(self, path):
        user_cfg = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
            except Exception as e:
                messagebox.showwarning("Config", f"Failed to parse {path}.\n{e}")
        self.cfg = merge_defaults(user_cfg, self.defaults)

    # ---------- populate ----------
    def populate_from_config(self):
        # Left column dynamic rebuild
        for child in self.left_inner_frame.winfo_children():
            child.destroy()
        self.entry_vars.clear()
        for section in SECTIONS:
            box = ttk.LabelFrame(self.left_inner_frame, text=section)
            box.pack(fill=tk.X, padx=6, pady=6)
            grid_row = 0
            self.entry_vars.setdefault(section, {})
            for key in sorted(self.cfg.get(section, {}).keys()):
                sv = tk.StringVar(value=str(self.cfg[section][key]))
                self.entry_vars[section][key] = sv
                ttk.Label(box, text=key).grid(row=grid_row, column=0, sticky="w", padx=4, pady=2)
                e = ttk.Entry(box, textvariable=sv, width=12)
                e.grid(row=grid_row, column=1, sticky="w", padx=4, pady=2)
                grid_row += 1

        # Middle weights grid
        for child in self.weights_frame.winfo_children():
            child.destroy()
        self.weight_vars.clear()
        weights = self.cfg.get("params", {}).get("weights", {})
        row = 0
        for varname in sorted(weights.keys()):
            ttk.Label(self.weights_frame, text=varname, width=12).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            for j, wkey in enumerate(["env","int","reg","nut","decay"], start=1):
                sv = tk.StringVar(value=str(weights[varname].get(wkey, 0.0)))
                self.weight_vars[(varname, wkey)] = sv
                ttk.Entry(self.weights_frame, textvariable=sv, width=8).grid(row=row, column=j, sticky="w", padx=4, pady=2)
            row += 1

        # Right run options
        run = self.cfg.get("run", {})
        self.run_steps.set(str(run.get("steps", 200)))
        self.run_name.set(run.get("agent_name", "Athena"))
        self.data_dir.set(run.get("data_dir", "data"))
        self.log_csv.set(bool(run.get("log_csv", True)))

    # ---------- collect ----------
    def collect_to_config(self):
        cfg = {"run": {}, "params": {"weights": {}}}
        # left sections
        for section, sub in self.entry_vars.items():
            cfg[section] = {}
            for key, sv in sub.items():
                raw = sv.get().strip()
                if raw == "": continue
                try: val = float(raw)
                except: val = 0.0
                if not is_open_range(section, key):
                    val = clamp01(val)
                cfg[section][key] = val

        # noise_std keep existing default unless we expose UI for it
        cfg["params"]["noise_std"] = float(self.cfg.get("params", {}).get("noise_std", 0.01))

        # weights
        weights = {}
        for (varname, wkey), sv in self.weight_vars.items():
            weights.setdefault(varname, {})
            try:
                weights[varname][wkey] = float(sv.get().strip())
            except:
                weights[varname][wkey] = 0.0
        cfg["params"]["weights"] = weights

        # run options
        try:
            steps = int(self.run_steps.get().strip())
        except:
            steps = 200
        cfg["run"]["steps"] = max(1, steps)
        cfg["run"]["agent_name"] = self.run_name.get().strip() or "Agent"
        cfg["run"]["data_dir"] = self.data_dir.get().strip() or "data"
        cfg["run"]["log_csv"] = bool(self.log_csv.get())

        # merge with defaults to keep anything not exposed
        return merge_defaults(cfg, self.defaults)

    # ---------- file ops ----------
    def choose_config_file(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON files","*.json")],
                                            initialdir=app_dir(),
                                            initialfile="cfss_config.json")
        if path: self.config_path.set(path)

    def load_config_btn(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files","*.json")],
                                          initialdir=app_dir())
        if not path: return
        self.config_path.set(path)
        self._load_cfg(path)
        self.populate_from_config()
        self.log(f"Loaded: {path}\n")

    def save_config_btn(self):
        cfg = self.collect_to_config()
        path = self.config_path.get() or default_config_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            self.cfg = cfg
            self.log(f"Saved config to {path}\n")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def choose_data_dir(self):
        d = filedialog.askdirectory(initialdir=app_dir())
        if d: self.data_dir.set(d)

    # ---------- run ----------
    def run_simulation(self):
        # Save current UI -> config file
        self.save_config_btn()
        cfg_path = self.config_path.get() or default_config_path()
        exe = sys.executable or "py"
        agent_py = os.path.join(app_dir(), "agent.py")
        if not os.path.exists(agent_py):
            messagebox.showerror("Agent Missing", f"agent.py not found at {agent_py}")
            return

        # ensure data dir exists
        os.makedirs(self.data_dir.get(), exist_ok=True)

        cmd = [exe, agent_py, "--config", cfg_path, "--no-pause", "--plot"]
        self.log(f"Running: {' '.join(cmd)}\n")
        self.run_btn.config(state="disabled")

        def worker():
            try:
                proc = subprocess.Popen(
                    cmd, cwd=app_dir(),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                self.proc = proc
                for line in proc.stdout:
                    self.output_queue.put(line)
                rc = proc.wait()
                self.output_queue.put(f"\n[Process exited {rc}]\n")
            except Exception as e:
                self.output_queue.put(f"\n[ERROR] {e}\n")
            finally:
                self.proc = None
                self.run_btn.config(state="normal")
                # refresh data list after run
                self.refresh_data_list()

        threading.Thread(target=worker, daemon=True).start()

    # ---------- console ----------
    def log(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    def _drain_output_queue(self):
        try:
            while True:
                line = self.output_queue.get_nowait()
                self.log(line)
        except queue.Empty:
            pass
        self.after(100, self._drain_output_queue)

    # ---------- data viewer ----------
    def refresh_data_list(self):
        self.csv_list.delete(0, tk.END)
        d = self.data_dir.get().strip() or "data"
        pattern = os.path.join(d, "*.csv")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for f in files:
            self.csv_list.insert(tk.END, os.path.basename(f))
        if files:
            self.csv_list.selection_clear(0, tk.END)
            self.csv_list.selection_set(0)
            self.preview_selected_csv()

    def selected_csv_path(self):
        sel = self.csv_list.curselection()
        if not sel:
            return None
        fname = self.csv_list.get(sel[0])
        return os.path.join(self.data_dir.get().strip() or "data", fname)

    def open_selected_csv(self):
        path = self.selected_csv_path()
        if not path: return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # default app
            elif sys.platform == "darwin":
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Open", str(e))

    def preview_selected_csv(self):
        path = self.selected_csv_path()
        if not path or not os.path.exists(path):
            return
        # clear preview
        for c in self.preview.get_children():
            self.preview.delete(c)
        for col in self.preview["columns"]:
            self.preview.heading(col, text="")
        self.preview["columns"] = ()

        # read first N rows
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            try:
                header = next(rdr)
            except StopIteration:
                return
            for i, row in enumerate(rdr):
                rows.append(row)
                if i >= 200: break

        # set columns
        self.preview["columns"] = header
        for h in header:
            self.preview.heading(h, text=h)
            self.preview.column(h, width=120, anchor="w")

        for row in rows:
            self.preview.insert("", "end", values=row)

if __name__ == "__main__":
    app = CFSSGui()
    app.mainloop()
