import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from backend.wing_model import analyze_wing

st.set_page_config(page_title="Wing Analyzer", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Parameters
st.sidebar.title("Wing Configuration")

# Presets
presets = {
    "General Aviation": {"span": 11.0, "ar": 7.2, "taper": 0.45, "sweep": 3.0, "alt": 2000.0, "vel": 55.0, "thick": 0.12, "cam": 0.02},
    "Commercial Jet": {"span": 60.0, "ar": 9.5, "taper": 0.28, "sweep": 28.0, "alt": 11000.0, "vel": 240.0, "thick": 0.11, "cam": 0.01},
    "Fighter Jet": {"span": 9.0, "ar": 2.8, "taper": 0.25, "sweep": 35.0, "alt": 8000.0, "vel": 280.0, "thick": 0.08, "cam": 0.00},
    "High AR Glider": {"span": 20.0, "ar": 22.0, "taper": 0.55, "sweep": 1.0, "alt": 1500.0, "vel": 28.0, "thick": 0.14, "cam": 0.03}
}

selected_preset = st.sidebar.selectbox("Select Preset", list(presets.keys()))
p = presets[selected_preset]

span = st.sidebar.slider("Wingspan (m)", 4.0, 60.0, p["span"], 0.5)
ar = st.sidebar.slider("Aspect Ratio", 2.0, 25.0, p["ar"], 0.1)
taper = st.sidebar.slider("Taper Ratio", 0.1, 1.0, p["taper"], 0.01)
sweep = st.sidebar.slider("Sweep Angle (deg)", 0.0, 60.0, p["sweep"], 0.5)
alt = st.sidebar.slider("Altitude (m)", 0, 12000, int(p["alt"]), 100)
vel = st.sidebar.slider("Cruise Speed (m/s)", 20, 400, int(p["vel"]), 1)
thick = st.sidebar.slider("Airfoil Thickness", 0.06, 0.24, p["thick"], 0.01)
cam = st.sidebar.slider("Camber", 0.00, 0.09, p["cam"], 0.01)

# Main Page Header
st.title("Wing Analyzer")
st.caption("Wing Analysis using Physics informed Machine Learning Model by Garige Surya Prakash and Kolanu Lokesh")

# Analysis Logic
if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Analyzing aerodynamic performance..."):
        inputs = {
            "wing_type": selected_preset,
            "span": span,
            "ar": ar,
            "taper": taper,
            "sweep_deg": sweep,
            "altitude": alt,
            "velocity": vel,
            "thickness": thick,
            "camber": cam
        }
        
        data = analyze_wing(inputs)
        s = data["summary"]
        
        # Metrics Strip
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max efficiency (L/D)", f"{s['best_LD']:.1f}")
        col2.metric("Optimal Lift (CL)", f"{s['best_CL']:.3f}")
        col3.metric("Structural Moment (RBM)", f"{s['RBM_Nm']/1000:.1f} kNm")
        col4.metric("Stall AoA (Deg)", f"{s['stall_aoa']:.1f}°")
        
        st.info(f"**Recommendation:** {s['recommendation']}")

        # Charts Section
        tab_physics, tab_geometry = st.tabs(["Aerodynamic Analysis", "3D Visual Model"])
        
        with tab_physics:
            c1, c2 = st.columns(2)
            
            # Polar
            fig_polar = px.line(data["polar"], x="CD", y="CL", title="Wing Polars (CL / CD)")
            fig_polar.update_traces(line_color='#1D4ED8', line_width=3)
            c1.plotly_chart(fig_polar, use_container_width=True)
            
            # Cl Distribution at Best AoA
            best_p = next(p for p in data["polar"] if p["aoa"] == s["best_aoa"])
            fig_cl_dist = px.area(x=data["y_coords"], y=best_p["cl_dist"], title="Spanwise Loading (CL @ Best AoA)")
            fig_cl_dist.update_traces(line_color='#1D4ED8')
            c2.plotly_chart(fig_cl_dist, use_container_width=True)
            
            c3, c4 = st.columns(2)
            # Stall Margin
            fig_stall = go.Figure()
            fig_stall.add_trace(go.Scatter(x=data["y_coords"], y=data["cl_max_dist"], name="Cl Max", line=dict(dash='dash', color='#059669')))
            fig_stall.add_trace(go.Scatter(x=data["y_coords"], y=best_p["cl_dist"], name="Actual Cl", fill="tonexty", line=dict(color='#DC2626')))
            fig_stall.update_layout(title="Stall Margin Distribution", template="plotly_white")
            c3.plotly_chart(fig_stall, use_container_width=True)
            
            # Drag Budget
            fig_drag = px.bar(x=list(data["drag_budget"].keys()), y=[v*10000 for v in data["drag_budget"].values()], 
                            title="Total Drag Budget (Counts)", labels={'x': 'Drag Source', 'y': 'Drag Counts (x10^4)'})
            fig_drag.update_traces(marker_color='#1D4ED8')
            c4.plotly_chart(fig_drag, use_container_width=True)

            # Performance Curves Row
            c5, c6 = st.columns(2)
            fig_lift = px.line(data["polar"], x="aoa", y="CL", title="Lift Curve (CL vs α)")
            c5.plotly_chart(fig_lift, use_container_width=True)
            fig_drag_curve = px.line(data["polar"], x="aoa", y="CD", title="Drag Curve (CD vs α)")
            c6.plotly_chart(fig_drag_curve, use_container_width=True)

            c7, c8 = st.columns(2)
            fig_ld = px.line(data["polar"], x="aoa", y="LD", title="Efficiency (L/D vs α)")
            c7.plotly_chart(fig_ld, use_container_width=True)
            
            if "Cm" in data["polar"][0]:
                fig_cm = px.line(data["polar"], x="aoa", y="Cm", title="Pitching Moment (Cm vs α)")
                c8.plotly_chart(fig_cm, use_container_width=True)

        with tab_geometry:
            g = data["geometry"]
            fig_3d = go.Figure()
            fig_3d.add_trace(go.Surface(x=g["X"], y=g["Y"], z=g["Z_top"], colorscale=[[0, '#eff6ff'], [1, '#1d4ed8']], showscale=False, name="Top"))
            fig_3d.add_trace(go.Surface(x=g["X"], y=g["Y"], z=g["Z_bot"], colorscale=[[0, '#f8fafc'], [1, '#94a3b8']], showscale=False, opacity=0.5, name="Bottom"))
            
            # LE/TE lines
            fig_3d.add_trace(go.Scatter3d(x=g["le_x"], y=g["y_st"], z=[0]*len(g["y_st"]), mode='lines', line=dict(color='black', width=5), name="LE"))
            te_x = [le + c for le, c in zip(g["le_x"], g["chord"])]
            fig_3d.add_trace(go.Scatter3d(x=te_x, y=g["y_st"], z=[0]*len(g["y_st"]), mode='lines', line=dict(color='#1e3a8a', width=4), name="TE"))
            
            fig_3d.update_layout(title="3D Wing Surface Model", scene=dict(aspectratio=dict(x=1, y=4, z=0.5)), height=700)
            st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("👋 Select parameters in the sidebar and click RUN ANALYSIS to start.")
