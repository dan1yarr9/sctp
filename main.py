"""
MILP және CP-SAT алгоритмдері арқылы мектеп сабақ кестесін комбинаторлық оңтайландыру.
Бастапқы деректер: 'data/schedule_data.json', 'data/current_schedule.xlsx' және 'data/survey_data.csv'.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- ЖОЛДАР МЕН ҚАЛТАЛАР ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DEFAULT_DIFFICULTY_PATTERNS = [
    (["алгебра", "математика", "анализ", "math"], 11),
    (["геометр"], 10),
    (["физик", "химия", "physics", "chemistry"], 10),
    (["информ", "цифр", "it", "компьютер"], 9),
    (["шет т", "ағылшын", "english", "неміс", "француз", "иностранный"], 9),
    (["қазақ т", "орыс т", "русский яз", "родной яз", "ана т", "тіл"], 8),
    (["биолог", "географ", "жаратылыс"], 7),
    (["тарих", "история", "құқық", "право"], 6),
    (["әдебиет", "литератур", "reading"], 5),
    (["жаһандық", "құзырет", "глобал"], 4),
    (["өзін-өзі", "аәд", "аәтд", "нвп", "технолог", "кәсіп"], 3),
    (["көркем", "еңбек", "музыка", "труд", "өнер", "сызу", "графика"], 2),
    (["дене", "спорт", "физкульт", "физра", "пэ", "pe"], 1),
]


def calculate_cronbach_alpha(df: pd.DataFrame) -> float:
    k = df.shape[1]
    if k <= 1:
        return 0.0
    item_variances = df.var(axis=0, ddof=1).sum()
    total_score_variance = df.sum(axis=1).var(ddof=1)
    if total_score_variance == 0:
        return 0.0
    return float((k / (k - 1)) * (1.0 - (item_variances / total_score_variance)))


def calculate_p_value(t_stat: float) -> float:
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


def load_and_process_survey(csv_path: Path):
    if not csv_path.exists():
        print(f"[ҚАТЕ] Сауалнаманың CSV файлы табылмады: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    clean_df = df.iloc[:, :6].apply(pd.to_numeric, errors="coerce").dropna()
    n_resp = len(clean_df)

    if n_resp == 0:
        print("[ҚАТЕ] survey_data.csv ішінде жарамды сандық дерек жоқ.")
        sys.exit(1)

    cronbach_a = calculate_cronbach_alpha(clean_df)
    a_mean, a_std, w_hat = {}, {}, {}
    summary_rows = []

    for idx in range(6):
        q_num = idx + 1
        col = clean_df.iloc[:, idx]
        mean_val = float(col.mean())
        std_val = float(col.std(ddof=1))
        sem_val = std_val / math.sqrt(n_resp)
        ci_low = round(mean_val - 1.96 * sem_val, 3)
        ci_high = round(mean_val + 1.96 * sem_val, 3)
        norm_w = (mean_val - 1.0) / 4.0

        t_stat = (mean_val - 3.0) / sem_val if sem_val > 0 else 0.0
        p_val = calculate_p_value(t_stat)

        a_mean[q_num] = mean_val
        a_std[q_num] = std_val
        w_hat[q_num] = norm_w

        p_str = "< 0.001 (***)" if p_val < 0.001 else f"{p_val:.4f}"

        summary_rows.append({
            "Сұрақ": f"№{q_num}",
            "Орташа балл (ā)": round(mean_val, 3),
            "Ауытқу (σ)": round(std_val, 3),
            "SEM": round(sem_val, 3),
            "95% Сенім аралығы": f"[{ci_low}; {ci_high}]",
            "t-статистика": round(t_stat, 2),
            "p-мәні": p_str,
            "Норм. салмақ (ŵ)": round(norm_w, 4),
        })

    return n_resp, a_mean, a_std, w_hat, cronbach_a, pd.DataFrame(summary_rows)


def calibrate_parameters(w_hat: dict, base_k: list):
    w1_hat = w_hat.get(1, 0.8)
    w2_hat = w_hat.get(2, 0.8)
    w6_hat = w_hat.get(6, 0.8)

    w_crit = (round(w6_hat, 4), 1.0, round(w1_hat, 4))

    mu4_5 = round(1.0 + 0.25 * w2_hat, 4)
    mu6_7 = round(1.0 + 0.50 * w2_hat, 4)
    mu_raw = (1.0, 1.0, 1.0, mu4_5, mu4_5, mu6_7, mu6_7)

    delta = 0.05 * w6_hat
    k1 = round(base_k[0] - delta / 2.0, 5)
    k2 = round(base_k[1] + delta / 2.0, 5)
    k3 = round(base_k[2] + delta / 2.0, 5)
    k4 = base_k[3]
    k5 = k1
    k_raw = (k1, k2, k3, k4, k5)

    return w_crit, mu_raw, k_raw


def get_subject_difficulty(subject: str) -> int:
    s_clean = subject.lower().replace(".", "").strip()
    for patterns, score in DEFAULT_DIFFICULTY_PATTERNS:
        if any(p in s_clean for p in patterns):
            return score
    return 5


def load_all_data():
    json_path = DATA_DIR / "schedule_data.json"
    excel_path = DATA_DIR / "current_schedule.xlsx"

    cfg = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[МӘЛІМЕТ] Баптаулар JSON файлынан сәтті жүктелді: {json_path.name}")
    else:
        print(f"[ЕСКЕРТУ] JSON табылмады. Деректер толықтай Excel арқылы жиналады.")

    excel_grid = {}
    if excel_path.exists():
        xls = pd.ExcelFile(excel_path)
        for sheet in xls.sheet_names:
            if str(sheet).startswith("_"):
                continue
            df = pd.read_excel(xls, sheet_name=sheet)
            first_col = str(df.columns[0]).strip().lower()
            if first_col in ["слот", "сабақ", "сабак", "урок", "№", "index", "slot", "p"] or df.iloc[:, 0].dtype in ['int64', 'float64']:
                df = df.set_index(df.columns[0])

            c_name = str(sheet).strip()
            excel_grid[c_name] = {}
            for d in df.columns:
                day_name = str(d).strip()
                excel_grid[c_name][day_name] = {
                    p_idx: ("" if pd.isna(v) else str(v).strip())
                    for p_idx, v in enumerate(df[d].tolist(), start=1)
                }

    classes = cfg.get("classes", list(excel_grid.keys()) or ["11А", "11Б", "11В"])
    days = cfg.get("days", ["Дс", "Сс", "Ср", "Бс", "Жм"])
    slots = cfg.get("slots", [1, 2, 3, 4, 5, 6, 7])

    current_grid = cfg.get("current_grid") or excel_grid
    if not current_grid:
        current_grid = {c: {d: {p: "" for p in slots} for d in days} for c in classes}
    else:
        for c in classes:
            if c not in current_grid:
                current_grid[c] = {}
            for d in days:
                if d not in current_grid[c]:
                    current_grid[c][d] = {}
                for p in slots:
                    if p not in current_grid[c][d]:
                        current_grid[c][d][p] = ""

    hours = cfg.get("curriculum_hours")
    if not hours:
        hours = {c: {} for c in classes}
        for c in classes:
            for d in days:
                for p in slots:
                    s = current_grid.get(c, {}).get(d, {}).get(p, "")
                    if s:
                        hours[c][s] = hours[c].get(s, 0) + 1

    diff_scores = cfg.get("difficulty_scores", {})
    all_subjects = sorted(list(set(s for c in classes for s in hours.get(c, {}).keys())))
    for s in all_subjects:
        if s not in diff_scores:
            diff_scores[s] = get_subject_difficulty(s)

    hard_subjects = cfg.get("hard_subjects", [s for s in all_subjects if diff_scores[s] >= 8])
    
    # Егер пән special_rooms ішінде арнайы шектелмесе, кабинет тапшылығы болмайды (лимит = сыныптар саны)
    special_rooms = cfg.get("special_rooms", {})
    for s in all_subjects:
        if s not in special_rooms:
            special_rooms[s] = len(classes)

    h_max = cfg.get("h_max_hard_per_day", {c: 4 for c in classes})
    frozen = [tuple(item) for item in cfg.get("frozen_slots", [])]

    params = cfg.get("parameters", {
        "lambda": 100,
        "max_time_sec": 30,
        "random_seed": 1,
        "soft_weight": 1000000,
        "base_k": [0.15, 0.25, 0.25, 0.20, 0.15],
    })

    return {
        "classes": classes,
        "days": days,
        "slots": slots,
        "parameters": params,
        "difficulty_scores": diff_scores,
        "hard_subjects": hard_subjects,
        "special_rooms": special_rooms,
        "h_max_hard_per_day": h_max,
        "curriculum_hours": hours,
        "current_grid": current_grid,
        "frozen_slots": frozen,
    }


# --- ДЕРЕКТЕРДІ ДАЙЫНДАУ ---
SCHEDULE_CFG = load_all_data()
N_RESP, A_MEAN, A_STD, W_HAT, CRONBACH_ALPHA, SURVEY_SUMMARY_DF = load_and_process_survey(DATA_DIR / "survey_data.csv")

CLASSES = SCHEDULE_CFG["classes"]
DAYS = SCHEDULE_CFG["days"]
SLOTS = SCHEDULE_CFG["slots"]
PARAMS = SCHEDULE_CFG["parameters"]
LAMBDA = PARAMS["lambda"]
MAX_TIME = PARAMS["max_time_sec"]
SEED = PARAMS["random_seed"]
SOFT_W = PARAMS["soft_weight"]

L = SCHEDULE_CFG["difficulty_scores"]
SUBJECTS = list(L.keys())
S_HARD = set(SCHEDULE_CFG["hard_subjects"])
R_CAB = SCHEDULE_CFG["special_rooms"]
H_MAX = SCHEDULE_CFG["h_max_hard_per_day"]
HOURS = SCHEDULE_CFG["curriculum_hours"]
CURRENT_GRID = SCHEDULE_CFG["current_grid"]
FROZEN = [tuple(item) for item in SCHEDULE_CFG["frozen_slots"]]

W_CRIT, MU_RAW, K_RAW = calibrate_parameters(W_HAT, PARAMS["base_k"])

STATUSES = {
    cp_model.OPTIMAL: "OPTIMAL (Оңтайлы)",
    cp_model.FEASIBLE: "FEASIBLE (Жарамды)",
    cp_model.INFEASIBLE: "INFEASIBLE (Шешімсіз)",
    cp_model.UNKNOWN: "UNKNOWN (Белгісіз)",
    cp_model.MODEL_INVALID: "MODEL_INVALID (Қате модель)",
}

CI = {c: i for i, c in enumerate(CLASSES)}
SI = {s: i for i, s in enumerate(SUBJECTS)}
DI = {d: i for i, d in enumerate(DAYS)}


def hour(c, s):
    return HOURS.get(c, {}).get(s, 0)


def n_max(s):
    return 1 if s in S_HARD else 2


def k_tilde():
    exact = [LAMBDA * v for v in K_RAW]
    ktil = [round(v) for v in exact]
    if len(ktil) == 5:
        ktil[4] = ktil[0]
    rem = LAMBDA - sum(ktil)
    order = sorted((1, 2), key=lambda i: exact[i] - ktil[i], reverse=True)
    step = 1 if rem > 0 else -1
    i = 0
    while rem != 0:
        ktil[order[i % len(order)]] += step
        rem -= step
        i += 1
    return ktil


def m_slots():
    return [round(LAMBDA * (mu - 1)) for mu in MU_RAW]


def cell(c, d, p):
    return CURRENT_GRID.get(c, {}).get(d, {}).get(p, "")


def active(c):
    return [s for s in SUBJECTS if hour(c, s) > 0]


def subject_caps():
    caps = {}
    for s in SUBJECTS:
        caps[s] = R_CAB.get(s, len(CLASSES))
    return caps


def metrics_of(grid):
    ktil = k_tilde()
    load0 = {c: sum(hour(c, s) * L.get(s, 0) for s in SUBJECTS) for c in CLASSES}
    dev = late_hard = consec = windows = 0
    for c in CLASSES:
        for di, d in enumerate(DAYS):
            w, occ, hard = 0, [], []
            for p in SLOTS:
                s = grid.get(c, {}).get(d, {}).get(p, "")
                if s:
                    occ.append(p)
                    w += L.get(s, 0)
                    hard.append(s in S_HARD)
                    if s in S_HARD and p >= 6:
                        late_hard += 1
                else:
                    hard.append(False)
            dev += abs(w - load0[c] * ktil[di] / LAMBDA)
            consec += sum(hard[i] and hard[i + 1] for i in range(len(hard) - 1))
            if occ:
                windows += occ[-1] - occ[0] + 1 - len(occ)
    return {"dev": dev, "late_hard": late_hard, "consec": consec, "windows": windows}


def current_violations():
    found = []
    for s, r in R_CAB.items():
        for d in DAYS:
            for p in SLOTS:
                n = sum(1 for c in CLASSES if cell(c, d, p) == s)
                if n > r:
                    found.append(f"Кабинет тапшылығы: {s} пәні {d} күні {p}-сабақта {n} сыныпта қойылған (Лимит R={r})")
    for c in CLASSES:
        for d in DAYS:
            cnt = Counter(cell(c, d, p) for p in SLOTS if cell(c, d, p))
            for s, k in cnt.items():
                if k > n_max(s):
                    found.append(f"Күндік қайталану: {c}, {d}, {s} пәні {k} рет қойылған (Рұқсат N_max={n_max(s)})")
            n_hard = sum(1 for p in SLOTS if cell(c, d, p) in S_HARD)
            if n_hard > H_MAX.get(c, 4):
                found.append(f"Күрделі пәндер лимиті: {c}, {d} күні {n_hard} ауыр пән бар (Лимит H_max={H_MAX.get(c, 4)})")
    return found


def precheck_messages():
    msgs, ktil = [], k_tilde()
    if sum(ktil) != 100:
        msgs.append(f"Σ K̃_d = {sum(ktil)}, міндетті мән: 100")
    for c in CLASSES:
        total = sum(hour(c, s) for s in SUBJECTS)
        if not 20 <= total <= 40:
            msgs.append(f"{c}: Апталық сағат қосындысы Σ h = {total} (талап: 20..40)")
    return msgs


def gap_pct(solver, st):
    if st == cp_model.OPTIMAL:
        return 0.0
    if st not in (cp_model.FEASIBLE, cp_model.UNKNOWN):
        return None
    obj = solver.ObjectiveValue()
    return 100.0 * abs(obj - solver.BestObjectiveBound()) / max(1.0, abs(obj))


def extract_grid(solver, x):
    grid = {c: {d: {p: "" for p in SLOTS} for d in DAYS} for c in CLASSES}
    for (c, s, d, p), var in x.items():
        if solver.Value(var) == 1:
            grid[c][d][p] = s
    return grid


def component_values(grid):
    ktil, mvec = k_tilde(), m_slots()
    load0 = {c: sum(hour(c, s) * L.get(s, 0) for s in SUBJECTS) for c in CLASSES}
    sum_y = late = consec = 0
    for c in CLASSES:
        for di, d in enumerate(DAYS):
            lam_w, hard = 0, []
            for p in SLOTS:
                s = grid.get(c, {}).get(d, {}).get(p, "")
                if s:
                    s_score = L.get(s, 0)
                    lam_w += LAMBDA * s_score
                    late += mvec[p - 1] * s_score
                    hard.append(s in S_HARD)
                else:
                    hard.append(False)
            sum_y += abs(lam_w - load0[c] * ktil[di])
            consec += sum(hard[i] and hard[i + 1] for i in range(len(hard) - 1))
    return sum_y, late, consec


def empty_result(st, elapsed, solver):
    return {
        "status": st, "name": STATUSES.get(st, str(st)), "time": elapsed,
        "gap": gap_pct(solver, st), "grid": None, "sum_y": None, "late": None,
        "consec": None, "obj": None, "solver": "CP-SAT",
    }


def apply_hints(model, x, z, grid):
    if grid is None:
        return
    for (c, s, d, p), var in x.items():
        model.AddHint(var, int(grid.get(c, {}).get(d, {}).get(p) == s))
    for c in CLASSES:
        for d in DAYS:
            hard = [grid.get(c, {}).get(d, {}).get(p) in S_HARD for p in SLOTS]
            for p in range(1, len(SLOTS)):
                model.AddHint(z[c, d, p], int(hard[p - 1] and hard[p]))


def build_solve(alpha, bounds, soft, hint_grid=None):
    model = cp_model.CpModel()
    ktil, mvec, caps = k_tilde(), m_slots(), subject_caps()
    x, y, z, occ, slack = {}, {}, {}, {}, []

    for c in CLASSES:
        for s in active(c):
            for d in DAYS:
                for p in SLOTS:
                    x[c, s, d, p] = model.NewBoolVar(f"x_{CI[c]}_{SI[s]}_{DI[d]}_{p}")
        for d in DAYS:
            y[c, d] = model.NewIntVar(0, 200000, f"y_{CI[c]}_{DI[d]}")
            for p in SLOTS:
                occ[c, d, p] = model.NewBoolVar(f"o_{CI[c]}_{DI[d]}_{p}")
            for p in range(1, len(SLOTS)):
                z[c, d, p] = model.NewBoolVar(f"z_{CI[c]}_{DI[d]}_{p}")

    # Әр ұяшықта ең көбі 1 сабақ
    for c in CLASSES:
        for d in DAYS:
            for p in SLOTS:
                terms = [x[c, s, d, p] for s in active(c)]
                model.Add(sum(terms) <= 1)
                model.Add(sum(terms) == occ[c, d, p])

    # Кабинеттер және мұғалімдер лимиті
    for s in SUBJECTS:
        users = [c for c in CLASSES if hour(c, s) > 0]
        if not users:
            continue
        for d in DAYS:
            for p in SLOTS:
                model.Add(sum(x[c, s, d, p] for c in users) <= caps[s])

    # Пәннің апталық сағатын толық орналастыру
    for c in CLASSES:
        for s in active(c):
            model.Add(sum(x[c, s, d, p] for d in DAYS for p in SLOTS) == hour(c, s))

    # Терезелерді (окна) болдырмау
    for c in CLASSES:
        for d in DAYS:
            for p in range(2, len(SLOTS) + 1):
                if soft:
                    sl = model.NewIntVar(0, 1, f"win_{c}_{d}_{p}")
                    model.Add(occ[c, d, p] <= occ[c, d, p - 1] + sl)
                    slack.append(sl)
                else:
                    model.Add(occ[c, d, p] <= occ[c, d, p - 1])

    # Күндік қайталану және жүктеме шектеулері
    for c in CLASSES:
        for s in active(c):
            for d in DAYS:
                model.Add(sum(x[c, s, d, p] for p in SLOTS) <= n_max(s))
        for d in DAYS:
            n_less = sum(occ[c, d, p] for p in SLOTS)
            n_hard = sum(x[c, s, d, p] for s in S_HARD if hour(c, s) > 0 for p in SLOTS)
            if soft:
                below = model.NewIntVar(0, 5, f"below_{c}_{d}")
                above = model.NewIntVar(0, 2, f"above_{c}_{d}")
                hover = model.NewIntVar(0, 7, f"hover_{c}_{d}")
                model.Add(n_less + below >= 4)
                model.Add(n_less - above <= 7)
                model.Add(n_hard <= H_MAX.get(c, 4) + hover)
                slack.extend([below, above, hover])
            else:
                model.Add(n_less >= 4)
                model.Add(n_less <= 7)
                model.Add(n_hard <= H_MAX.get(c, 4))

    # Бекітілген сабақтар (Frozen slots) - ТЕК КӨРСЕТІЛГЕН СЛОТТЫ 1 ҚЫЛУ
    for c, s, fd, fp in FROZEN:
        if c in CLASSES and s in active(c) and fd in DAYS and fp in SLOTS:
            model.Add(x[c, s, fd, fp] == 1)

    # Күндік когнитивті баланс (Ауытқу)
    load0 = {c: sum(hour(c, s) * L.get(s, 0) for s in SUBJECTS) for c in CLASSES}
    for c in CLASSES:
        for di, d in enumerate(DAYS):
            lam_w = sum(LAMBDA * L.get(s, 0) * x[c, s, d, p] for s in active(c) for p in SLOTS)
            lam_i = load0[c] * ktil[di]
            diff = model.NewIntVar(-200000, 200000, f"diff_{CI[c]}_{DI[d]}")
            model.Add(diff == lam_w - lam_i)
            model.AddAbsEquality(y[c, d], diff)

        for d in DAYS:
            for p in range(1, len(SLOTS)):
                hard_p = sum(x[c, s, d, p] for s in S_HARD if hour(c, s) > 0)
                hard_n = sum(x[c, s, d, p + 1] for s in S_HARD if hour(c, s) > 0)
                model.Add(z[c, d, p] >= hard_p + hard_n - 1)

    penalty_late = sum(
        mvec[p - 1] * L.get(s, 0) * x[c, s, d, p]
        for c in CLASSES for s in active(c) for d in DAYS for p in SLOTS
    )
    sum_y = sum(y.values())
    penalty_consec = sum(z.values())

    if bounds:
        if "sum_y" in bounds:
            model.Add(sum_y <= bounds["sum_y"])
        if "late" in bounds:
            model.Add(penalty_late <= bounds["late"])

    obj = alpha[0] * sum_y + alpha[1] * penalty_late + alpha[2] * penalty_consec
    if slack:
        obj = obj + SOFT_W * sum(slack)

    model.Minimize(obj)
    apply_hints(model, x, z, hint_grid)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = MAX_TIME
    solver.parameters.random_seed = SEED
    solver.parameters.num_search_workers = 8
    solver.parameters.repair_hint = True

    t0 = perf_counter()
    st = solver.Solve(model)
    elapsed = perf_counter() - t0

    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return empty_result(st, elapsed, solver)

    grid = extract_grid(solver, x)
    sy, late, consec = component_values(grid)
    return {
        "status": st, "name": STATUSES.get(st, str(st)), "time": elapsed,
        "gap": gap_pct(solver, st), "grid": grid, "sum_y": sy, "late": late,
        "consec": consec, "obj": solver.ObjectiveValue(), "solver": "CP-SAT",
    }


MILP_STATUS = None


def milp_backend():
    for name in ("CBC", "SCIP", "HIGHS"):
        slv = pywraplp.Solver.CreateSolver(name)
        if slv is not None:
            return slv, name
    return None, None


def milp_gap(solver, name):
    if "OPTIMAL" in name:
        return 0.0
    obj = solver.Objective().Value()
    bound = solver.Objective().BestBound()
    return 100.0 * abs(obj - bound) / max(1.0, abs(obj))


def extract_grid_lp(x):
    grid = {c: {d: {p: "" for p in SLOTS} for d in DAYS} for c in CLASSES}
    for (c, s, d, p), var in x.items():
        if var.solution_value() > 0.5:
            grid[c][d][p] = s
    return grid


def build_solve_milp(alpha, bounds, soft):
    global MILP_STATUS
    solver, backend = milp_backend()
    if solver is None:
        return {
            "status": None, "name": "ШЕШІЛМЕДІ (NOT_SOLVED)", "time": 0.0, "gap": None,
            "grid": None, "sum_y": None, "late": None, "consec": None,
            "obj": None, "solver": "—",
        }
    if MILP_STATUS is None:
        MILP_STATUS = {
            solver.OPTIMAL: "OPTIMAL (Оңтайлы)",
            solver.FEASIBLE: "FEASIBLE (Жарамды)",
            solver.INFEASIBLE: "INFEASIBLE (Шешімсіз)",
            solver.NOT_SOLVED: "NOT_SOLVED (Шешілмеді)",
            solver.ABNORMAL: "ABNORMAL (Қате)",
        }
    ktil, mvec, caps = k_tilde(), m_slots(), subject_caps()
    x, y, z, occ, slack = {}, {}, {}, {}, []

    for c in CLASSES:
        for s in active(c):
            for d in DAYS:
                for p in SLOTS:
                    x[c, s, d, p] = solver.BoolVar(f"x_{CI[c]}_{SI[s]}_{DI[d]}_{p}")
        for d in DAYS:
            y[c, d] = solver.IntVar(0, 200000, f"y_{CI[c]}_{DI[d]}")
            for p in SLOTS:
                occ[c, d, p] = solver.BoolVar(f"o_{CI[c]}_{DI[d]}_{p}")
            for p in range(1, len(SLOTS)):
                z[c, d, p] = solver.BoolVar(f"z_{CI[c]}_{DI[d]}_{p}")

    for c in CLASSES:
        for d in DAYS:
            for p in SLOTS:
                terms = [x[c, s, d, p] for s in active(c)]
                solver.Add(sum(terms) <= 1)
                solver.Add(sum(terms) == occ[c, d, p])

    for s in SUBJECTS:
        users = [c for c in CLASSES if hour(c, s) > 0]
        if not users:
            continue
        for d in DAYS:
            for p in SLOTS:
                solver.Add(sum(x[c, s, d, p] for c in users) <= caps[s])

    for c in CLASSES:
        for s in active(c):
            solver.Add(sum(x[c, s, d, p] for d in DAYS for p in SLOTS) == hour(c, s))

    for c in CLASSES:
        for d in DAYS:
            for p in range(2, len(SLOTS) + 1):
                if soft:
                    sl = solver.IntVar(0, 1, f"win_{c}_{d}_{p}")
                    solver.Add(occ[c, d, p] <= occ[c, d, p - 1] + sl)
                    slack.append(sl)
                else:
                    solver.Add(occ[c, d, p] <= occ[c, d, p - 1])

    for c in CLASSES:
        for s in active(c):
            for d in DAYS:
                solver.Add(sum(x[c, s, d, p] for p in SLOTS) <= n_max(s))
        for d in DAYS:
            n_less = sum(occ[c, d, p] for p in SLOTS)
            n_hard = sum(x[c, s, d, p] for s in S_HARD if hour(c, s) > 0 for p in SLOTS)
            if soft:
                below = solver.IntVar(0, 5, f"below_{c}_{d}")
                above = solver.IntVar(0, 2, f"above_{c}_{d}")
                hover = solver.IntVar(0, 7, f"hover_{c}_{d}")
                solver.Add(n_less + below >= 4)
                solver.Add(n_less - above <= 7)
                solver.Add(n_hard <= H_MAX.get(c, 4) + hover)
                slack.extend([below, above, hover])
            else:
                solver.Add(n_less >= 4)
                solver.Add(n_less <= 7)
                solver.Add(n_hard <= H_MAX.get(c, 4))

    for c, s, fd, fp in FROZEN:
        if c in CLASSES and s in active(c) and fd in DAYS and fp in SLOTS:
            solver.Add(x[c, s, fd, fp] == 1)

    load0 = {c: sum(hour(c, s) * L.get(s, 0) for s in SUBJECTS) for c in CLASSES}
    for c in CLASSES:
        for di, d in enumerate(DAYS):
            lam_w = sum(LAMBDA * L.get(s, 0) * x[c, s, d, p] for s in active(c) for p in SLOTS)
            lam_i = load0[c] * ktil[di]
            solver.Add(y[c, d] >= lam_w - lam_i)
            solver.Add(y[c, d] >= lam_i - lam_w)
        for d in DAYS:
            for p in range(1, len(SLOTS)):
                hard_p = sum(x[c, s, d, p] for s in S_HARD if hour(c, s) > 0)
                hard_n = sum(x[c, s, d, p + 1] for s in S_HARD if hour(c, s) > 0)
                solver.Add(z[c, d, p] >= hard_p + hard_n - 1)

    penalty_late = sum(
        mvec[p - 1] * L.get(s, 0) * x[c, s, d, p]
        for c in CLASSES for s in active(c) for d in DAYS for p in SLOTS
    )
    sum_y = sum(y.values())
    penalty_consec = sum(z.values())

    if bounds:
        if "sum_y" in bounds:
            solver.Add(sum_y <= bounds["sum_y"])
        if "late" in bounds:
            solver.Add(penalty_late <= bounds["late"])

    obj = alpha[0] * sum_y + alpha[1] * penalty_late + alpha[2] * penalty_consec
    if slack:
        obj = obj + SOFT_W * sum(slack)

    solver.Minimize(obj)
    solver.SetTimeLimit(30000)
    t0 = perf_counter()
    st = solver.Solve()
    elapsed = perf_counter() - t0
    name = MILP_STATUS.get(st, str(st))

    if st not in (solver.OPTIMAL, solver.FEASIBLE):
        return {
            "status": st, "name": name, "time": elapsed, "gap": None, "grid": None,
            "sum_y": None, "late": None, "consec": None, "obj": None, "solver": backend,
        }
    gap = milp_gap(solver, name)
    grid = extract_grid_lp(x)
    sy, late, consec = component_values(grid)
    return {
        "status": st, "name": name, "time": elapsed, "gap": gap, "grid": grid,
        "sum_y": sy, "late": late, "consec": consec,
        "obj": solver.Objective().Value(), "solver": backend,
    }


def run_lex(solve_fn, soft):
    r1 = solve_fn((1, 0, 0), None, soft)
    total = r1["time"]
    if r1["grid"] is None:
        out = dict(r1)
        out["time"] = total
        return out
    r2 = solve_fn((0, 1, 0), {"sum_y": r1["sum_y"]}, soft)
    total += r2["time"]
    if r2["grid"] is None:
        out = dict(r2)
        out["time"] = total
        return out
    r3 = solve_fn((0, 0, 1), {"sum_y": r1["sum_y"], "late": r2["late"]}, soft)
    total += r3["time"]
    out = dict(r3)
    out["time"] = total
    return out


def cpsat_plain(alpha, bounds, soft):
    return build_solve(alpha, bounds, soft, None)


def alphas_from_omega(om):
    omax = max(om)
    return tuple(max(1, round(LAMBDA * w * omax / max(1, oj))) for w, oj in zip(W_CRIT, om))


def csv_row(name, res, mets, alpha):
    gap = res["gap"]
    mets = mets or {"dev": None, "late_hard": None, "consec": None, "windows": None}
    return {
        "Шешкіш": res.get("solver") or "—",
        "Оңтайландыру режимі": name,
        "Мәртебесі": res["name"],
        "Уақыты (с)": round(res["time"], 4),
        "Алшақтық (Gap, %)": None if gap is None else round(gap, 4),
        "alpha_1": None if alpha is None else alpha[0],
        "alpha_2": None if alpha is None else alpha[1],
        "alpha_3": None if alpha is None else alpha[2],
        "Мінсіз профильден ауытқу": None if mets["dev"] is None else round(mets["dev"], 4),
        "6-7 сабақтағы күрделі пәндер": mets["late_hard"],
        "Қатар келген ауыр жұптар": mets["consec"],
        "Терезелер саны": mets["windows"],
    }


def write_outputs(cpsat_w, cpsat_lex, milp_w, milp_lex, cur_m, opt_m, rows, summary_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = OUTPUT_DIR / "schedule.xlsx"
    csv_path = OUTPUT_DIR / "results.csv"
    survey_path = OUTPUT_DIR / "survey_summary.csv"

    if cpsat_w["grid"] is not None:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for c in CLASSES:
                data = {d: [cpsat_w["grid"][c][d][p] for p in SLOTS] for d in DAYS}
                pd.DataFrame(data, index=[f"{p}-сабақ" for p in SLOTS]).to_excel(writer, sheet_name=c)

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(survey_path, index=False, encoding="utf-8-sig")

    print(f"\n[СӘТТІ АЯҚТАЛДЫ] Барлық нәтижелер мына қалтаға жазылды: {OUTPUT_DIR}")
    print(f" ├── 📊 Оңтайландырылған кесте (Excel): {excel_path.name}")
    print(f" ├── 📈 Салыстырмалы метрикалар (CSV): {csv_path.name}")
    print(f" └── 📋 Сауалнама талдауы (CSV): {survey_path.name}")


def main():
    print("=" * 70)
    print("  ОҚУШЫЛАРДЫҢ КОГНИТИВТІ ЖҮКТЕМЕСІН ЕСКЕРЕ ОТЫРЫП МЕКТЕП КЕСТЕСІН")
    print("      MILP ЖӘНЕ CP-SAT АЛГОРИТМДЕРІ АРҚЫЛЫ ОҢТАЙЛАНДЫРУ КЕШЕНІ")
    print("=" * 70)

    print(f"\n[1/5] Сауалнама деректерін өңдеу (N = {N_RESP} оқушы)...")
    print(SURVEY_SUMMARY_DF.to_string(index=False))
    print(f"\n      Психометриялық көрсеткіштер:")
    print(f"      ├── Кронбах альфасы: {CRONBACH_ALPHA:.4f}")
    print(f"      ├── Калибрленген нормативтік вектор K̃_d: {k_tilde()}")
    print(f"      ├── Мультипликаторлар M_p: {m_slots()}")
    print(f"      └── Критерийлер салмағы W_crit: {W_CRIT}")

    msgs = precheck_messages()
    if msgs:
        for m in msgs:
            print("[ВАЛИДАЦИЯ ҚАТЕСІ]:", m)
        return
    print("\n[2/5] Кіріс деректерді алдын ала тексеру сәтті өтті.")

    viol = current_violations()
    print("[3/5] Қолданыстағы кестедегі СанҚмН нормаларының бұзылуы:")
    print("\n".join("       - " + v for v in viol) if viol else "       - Норма бұзылулары тіркелмеді")

    cur_m = metrics_of(CURRENT_GRID)
    print("      Бастапқы метрикалар:", cur_m)

    print("\n[4/5] Сынақ оңтайландыруын жүргізу (Ω масштабын табу)...")
    soft = False
    trial = build_solve((1, 1, 1), None, soft, None)
    if trial["name"].startswith("INFEASIBLE"):
        print("      Қатаң режимде шешім жоқ. Жұмсақ (soft) шектеулер қосылды.")
        soft = True
        trial = build_solve((1, 1, 1), None, soft, None)

    if trial["grid"] is None:
        print("[ҚАТЕ] Шешім табылмады. Шешкіш мәртебесі:", trial["name"])
        return

    omega = (trial["sum_y"], trial["late"], trial["consec"])
    alpha = alphas_from_omega(omega)
    print(f"      Масштабтар: Ω = {omega} -> Мақсатты функция салмақтары: α = {alpha}")

    print("\n[5/5] Негізгі модельдерді есептеу...")
    weighted = cpsat_plain(alpha, None, soft)
    print(f"      [CP-SAT] Салмақтық: {weighted['name']} ({weighted['time']:.2f} с, gap = {weighted['gap']}%)")

    lex = run_lex(cpsat_plain, soft)
    print(f"      [CP-SAT] Лексикографикалық: {lex['name']} ({lex['time']:.2f} с, gap = {lex['gap']}%)")

    milp_w = build_solve_milp(alpha, None, soft)
    print(f"      [MILP]   Салмақтық: {milp_w['solver']} {milp_w['name']} ({milp_w['time']:.2f} с, gap = {milp_w['gap']}%)")

    milp_lex = run_lex(build_solve_milp, soft)
    print(f"      [MILP]   Лексикографикалық: {milp_lex['solver']} {milp_lex['name']} ({milp_lex['time']:.2f} с)")

    opt_m = metrics_of(weighted["grid"]) if weighted["grid"] is not None else cur_m
    w_m = metrics_of(weighted["grid"]) if weighted["grid"] is not None else None
    lex_m = metrics_of(lex["grid"]) if lex["grid"] is not None else None
    mw_m = metrics_of(milp_w["grid"]) if milp_w["grid"] is not None else None
    ml_m = metrics_of(milp_lex["grid"]) if milp_lex["grid"] is not None else None

    rows = [
        csv_row("Қолданыстағы кесте", {"name": "—", "time": 0.0, "gap": None, "solver": "—"}, cur_m, None),
        csv_row("CP-SAT (Салмақтық)", weighted, w_m, alpha),
        csv_row("CP-SAT (Лексикографиялық)", lex, lex_m, None),
        csv_row("MILP (Салмақтық)", milp_w, mw_m, alpha),
        csv_row("MILP (Лексикографиялық)", milp_lex, ml_m, None),
    ]

    write_outputs(weighted, lex, milp_w, milp_lex, cur_m, opt_m, rows, SURVEY_SUMMARY_DF)

    show = weighted["grid"]
    if show:
        print("\n" + "=" * 70)
        print("  ОҢТАЙЛАНДЫРЫЛҒАН САБАҚ КЕСТЕСІ (GOOGLE CP-SAT ШЕШІМІ)")
        print("=" * 70)
        for c in CLASSES:
            print(f"\n>>> Сынып: {c}")
            for d in DAYS:
                print(f"  {d:4s}: " + " | ".join(f"{show[c][d][p]:<18}" for p in SLOTS))


if __name__ == "__main__":
    main()