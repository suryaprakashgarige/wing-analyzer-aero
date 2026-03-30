import numpy as np
import pickle
import json
import os

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ── Load ML models ──
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "model_cl.pkl"), "rb") as f :
    m_cl = pickle.load(f)

with open(os.path.join(BASE, "model_cd.pkl"), "rb") as f :
    m_cd = pickle.load(f)

with open(os.path.join(BASE, "features.json")) as f :
    feature_info = json.load(f)

FEATURE_COLS = feature_info.get("features", [
    "Re", "alpha", "thickness", "thickness_loc", "camber", "camber_loc"
])

def predict_section_aero(Re, alpha, thickness, t_loc, camber, c_loc):
    X = np.array([[Re, alpha, thickness, t_loc, camber, c_loc]])
    return {
        "Cl": float(m_cl.predict(X)[0]),
        "Cd": float(abs(m_cd.predict(X)[0]))
    }

def isa_atmosphere(altitude):
    T   = 288.15 - 0.0065 * altitude
    rho = 1.225 * (T / 288.15) ** 4.256
    mu  = 1.789e-5 * (T / 288.15) ** 0.7
    return T, rho, mu

def run_llt(span, AR, taper, sweep_deg, velocity, altitude,
            thickness, camber, aoa_range):
    T, rho, mu = isa_atmosphere(altitude)
    S          = span ** 2 / AR
    root_chord = 2 * S / (span * (1 + taper))
    tip_chord  = root_chord * taper

    N      = 40
    theta  = np.linspace(np.pi/(2*N), np.pi*(1-1/(2*N)), N)
    y      = (span / 2) * np.cos(theta)
    y_norm = np.abs(y) / (span / 2)
    chord  = root_chord - (root_chord - tip_chord) * y_norm
    Re_sec = rho * velocity * chord / mu

    polars = []
    # Store distributions for the best L/D point

    for aoa in aoa_range:
        cls, cds = [], []
        for i in range(N):
            r  = predict_section_aero(Re_sec[i], aoa, thickness, 0.3, camber, 0.4)
            cls.append(r["Cl"])
            cds.append(r["Cd"])
        cls = np.array(cls); cds = np.array(cds)

        # LLT induced angle correction (New NumPy 2.0+ trapezoid)
        cl_avg    = _trapz(cls * chord, y) / (_trapz(chord, y) + 1e-9)
        e         = 1.78 * (1 - 0.045 * AR**0.68) - 0.64
        e         = max(0.65, min(e, 0.95))
        CDi       = cl_avg**2 / (np.pi * AR * e)
        CD0_sec   = float(_trapz(cds * chord, y) / (_trapz(chord, y) + 1e-9))
        CD_total  = CD0_sec + CDi
        LD        = cl_avg / CD_total if CD_total > 0 else 0

        Cm_ac = -np.pi * camber / 2        # moment about aerodynamic center
        Cm    = round(float(Cm_ac), 5)
        
        alpha_i_dist = np.degrees(cls / (np.pi * AR))   # LLT elliptic approx per station

        polars.append({
            "aoa":  round(float(aoa), 1),
            "CL":   round(float(cl_avg), 4),
            "CD":   round(float(CD_total), 5),
            "CDi":  round(float(CDi), 5),
            "CD0":  round(float(CD0_sec), 5),
            "LD":   round(float(LD), 2),
            "e":    round(float(e), 4),
            "cl_dist": cls.tolist(), # For spanwise plot
            "Cm": Cm,
            "alpha_i_dist": alpha_i_dist.tolist()
        })

    return polars, S, root_chord, tip_chord, Re_sec, y.tolist(), chord.tolist()

def build_3d_geometry(span, AR, taper, sweep_deg, N_span=50, N_chord=30, thickness=0.12):
    S          = span ** 2 / AR
    root_chord = 2 * S / (span * (1 + taper))
    tip_chord  = root_chord * taper
    sweep_rad  = np.radians(sweep_deg)

    y_st    = np.linspace(-span/2, span/2, N_span)
    y_norm  = np.abs(y_st) / (span/2)
    chord_s = root_chord - (root_chord - tip_chord) * y_norm
    le_x    = np.abs(y_st) * np.tan(sweep_rad)

    xi = np.linspace(0, 1, N_chord)
    t  = thickness
    zt = 5*t*(0.2969*np.sqrt(xi+1e-9) - 0.1260*xi
              - 0.3516*xi**2 + 0.2843*xi**3 - 0.1015*xi**4)

    X_top = []; Y_arr = []; Z_top = []; Z_bot = []
    for j in range(N_span):
        c  = chord_s[j]; le = le_x[j]
        X_top.append((le + xi * c).tolist())
        Y_arr.append([float(y_st[j])] * N_chord)
        Z_top.append((zt * c).tolist())
        Z_bot.append((-zt * c).tolist())

    return {
        "X": X_top, "Y": Y_arr,
        "Z_top": Z_top, "Z_bot": Z_bot,
        "le_x": le_x.tolist(), "y_st": y_st.tolist(),
        "chord": chord_s.tolist()
    }

def analyze_wing(params):
    span      = params["span"]
    AR        = params["ar"]
    taper     = params["taper"]
    sweep_deg = params["sweep_deg"]
    altitude  = params["altitude"]
    velocity  = params["velocity"]
    thickness = params["thickness"]
    camber    = params["camber"]
    wing_type = params["wing_type"]

    aoa_range = np.arange(-4, 16, 1)
    polars, S, root_c, tip_c, Re_sec, y_coords, chord_dist = run_llt(
        span, AR, taper, sweep_deg, velocity, altitude,
        thickness, camber, aoa_range
    )

    best = max(polars, key=lambda x: x["LD"])
    stall_aoa = 14.0
    for i in range(3, len(polars)):
        if polars[i]["CL"] < polars[i-1]["CL"] - 0.01:
            stall_aoa = polars[i]["aoa"]
            break

    # ── Stall Margin Analysis ──
    cl_max_dist = []
    for re_i in Re_sec:
        aoa_t = np.arange(-2, 18, 1)
        cls   = [predict_section_aero(re_i, a, thickness, 0.3, camber, 0.4)['Cl'] for a in aoa_t]
        cl_max_dist.append(max(cls))

    # ── Total Aircraft Drag Budget (Cell 17 Logic) ──
    fuse_length   = span * 0.72
    fuse_diameter = fuse_length / 8.0
    S_wet_fuse    = np.pi * fuse_diameter * fuse_length
    T, rho, mu    = isa_atmosphere(altitude)
    Re_fuse       = rho * velocity * fuse_length / mu
    Cf_fuse       = 0.455 / (np.log10(Re_fuse + 1)**2.58)
    f_ratio       = fuse_length / fuse_diameter
    FF_fuse       = 1 + 60/f_ratio**3 + f_ratio/400
    CD0_fuse      = Cf_fuse * FF_fuse * S_wet_fuse / (S + 1e-9)
    CD0_tail      = 0.25 * best["CD0"]
    CD0_inter     = 0.05 * (best["CD0"] + CD0_fuse + CD0_tail)

    drag_budget = {
        "Wing Parasite": round(best["CD0"], 5),
        "Fuselage Drag": round(CD0_fuse, 5),
        "Tail Drag":     round(CD0_tail, 5),
        "Interference":  round(CD0_inter, 5),
        "Induced Drag":  round(best["CDi"], 5)
    }

    # ── Root bending moment ──
    q    = 0.5 * rho * velocity**2
    L    = q * S * best["CL"]
    RBM  = round(L * span / 6, 1)

    rec = f"Wing type: {wing_type}. AR={AR:.1f}. Best L/D: {best['LD']:.1f}."
    if AR > 12: rec += " High AR glider — excellent cruise efficiency."
    if taper < 0.35: rec += " Low taper risk — monitor tip stall."

    geo = build_3d_geometry(span, AR, taper, sweep_deg, thickness=thickness)

    return {
        "polar": polars,
        "cl_max_dist": cl_max_dist, # NEW
        "y_coords": y_coords,       # NEW
        "drag_budget": drag_budget, # NEW
        "geometry": geo,
        "summary": {
            "wing_type":  wing_type,
            "best_LD":    best["LD"],
            "best_CL":    best["CL"],
            "best_CD":    best["CD"],
            "best_aoa":   best["aoa"],
            "stall_aoa":  stall_aoa,
            "span":       span,
            "area_m2":    round(S, 2),
            "root_chord": round(root_c, 3),
            "tip_chord":  round(tip_c, 3),
            "RBM_Nm":     RBM,
            "AR":         AR,
            "taper":      taper,
            "sweep_deg":  sweep_deg,
            "altitude_m": altitude,
            "velocity_ms":velocity,
            "recommendation": rec
        }
    }


