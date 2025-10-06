# app.py
"""
Bacterial Growth Pattern Predictor — Exhibition-ready
Features:
- Mode toggle: Normal Growth / Antibiotic Simulation
- Growth curve analyzer with phase detection
- Treated vs Untreated simulation
- Sample colony generator
- CSV & PDF export
- QR code generator
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import qrcode
import os

# Check PDF availability
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

 

# ---------- Helper functions ----------
def process_image_bytes(file_bytes, min_contour_area=50):
    arr = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0.0, 0
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    total_area = float(sum(cv2.contourArea(c) for c in kept))
    count = len(kept)
    cv2.drawContours(orig, kept, -1, (0,255,0), 2)
    disp_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    return disp_rgb, total_area, count

def detect_phase(prev_area, area, dt_hours):
    if area <= 0:
        return "No colony"
    if dt_hours <= 0:
        return "Lag" if area < 2000 else "Log"
    if prev_area <= 0:
        return "Lag" if area < 2000 else "Log"
    rel = ((area - prev_area)/prev_area)*100.0/dt_hours
    if rel <= -1.0: return "Decline"
    if rel < 5.0: return "Lag"
    if rel >= 20.0: return "Log"
    if abs(rel) < 5.0: return "Stationary"
    return "Log" if rel>0 else "Stationary"

def predict_next_area(times, areas):
    if len(times) < 1: return 1.0, 0.0
    x = np.array(times); y = np.array(areas)
    if len(x)==1: return x[-1]+1.0, float(y[-1])
    if np.allclose(x,x[0]):
        next_t = x[-1]+1.0
    else:
        next_t = float(x[-1]+np.median(np.diff(x)))
    try:
        a,b = np.polyfit(x,y,1)
        pred = float(a*next_t + b)
        if pred<0: pred=0.0
    except: pred = float(y[-1])
    return next_t, pred

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def make_pdf_report(title, df, fig_bytes, sample_images_bytes_list):
    if not PDF_AVAILABLE: return None
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,title,ln=True,align="C")
    pdf.ln(4)
    pdf.set_font("Arial",size=10)
    colw = 40

    # Show Group column if present
    if "Group" in df.columns:
        pdf.cell(20,8,"Day",1)
        pdf.cell(colw,8,"Time(h)",1)
        pdf.cell(colw,8,"Area(px)",1)
        pdf.cell(colw,8,"Phase",1)
        pdf.cell(colw,8,"Group",1)
        pdf.ln()
        for i,row in df.iterrows():
            pdf.cell(20,7,str(int(row["Day"])),1)
            pdf.cell(colw,7,f"{row['Time']:.2f}",1)
            pdf.cell(colw,7,f"{row['Area']:.1f}",1)
            pdf.cell(colw,7,str(row["Phase"]),1)
            pdf.cell(colw,7,str(row["Group"]),1)
            pdf.ln()
    else:
        pdf.cell(20,8,"Day",1)
        pdf.cell(colw,8,"Time(h)",1)
        pdf.cell(colw,8,"Area(px)",1)
        pdf.cell(colw,8,"Phase",1)
        pdf.ln()
        for i,row in df.iterrows():
            pdf.cell(20,7,str(int(row["Day"])),1)
            pdf.cell(colw,7,f"{row['Time']:.2f}",1)
            pdf.cell(colw,7,f"{row['Area']:.1f}",1)
            pdf.cell(colw,7,str(row["Phase"]),1)
            pdf.ln()
    pdf.add_page()
    if fig_bytes:
        fn="temp_plot.png"
        with open(fn,"wb") as f: f.write(fig_bytes)
        pdf.image(fn,x=10,w=190)
        try: os.remove(fn)
        except: pass
    for im_b in sample_images_bytes_list[:6]:
        fn="temp_img.png"
        with open(fn,"wb") as f: f.write(im_b)
        pdf.add_page()
        pdf.image(fn,x=20,w=170)
        try: os.remove(fn)
        except: pass
    out = BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

def make_qr_image_bytes(url,size=300):
    qr = qrcode.QRCode(box_size=10,border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black",back_color="white")
    bio = BytesIO()
    img = img.resize((size,size))
    img.save(bio,format="PNG")
    bio.seek(0)
    return bio.read()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Bacterial Growth Predictor", layout="wide", page_icon="🧬")
st.title("🧬 Bacterial Growth Pattern Predictor")

mode = st.radio("Select Mode", ["Normal Growth","Antibiotic Simulation"])

# ---------- NORMAL GROWTH ----------
if mode=="Normal Growth":
    st.header("Normal Growth Analyzer")
    uploaded = st.file_uploader("Upload images (ordered by time)", accept_multiple_files=True, type=["jpg","jpeg","png"])
    times=[]
    if uploaded:
        for i,f in enumerate(uploaded,start=1):
            t = st.number_input(f"Time (hours) for image {i}", value=float(i-1))
            times.append(float(t))
    else:
        if st.button("Generate sample colonies"):
            radii=[20,50,90,130]
            uploaded=[]
            for r in radii:
                img=np.ones((300,300,3),dtype=np.uint8)*255
                cv2.circle(img,(150,150),r,(0,0,0),-1)
                _,buf=cv2.imencode(".png",img)
                uploaded.append(BytesIO(buf.tobytes()))
            times=[0.0,1.0,2.0,3.0]
            st.experimental_rerun()
    if uploaded:
        min_area = st.slider("Min contour area (px)",10,2000,50)
        smoothing = st.slider("Smoothing sigma (0=off)",0.0,3.0,1.0,0.5)
        show_pred = st.checkbox("Show next-day prediction", value=True)
        areas=[]; counts=[]; images_disp=[]
        for f in uploaded:
            if hasattr(f,"read"): f.seek(0); b=f.read()
            else: b=f.getvalue()
            disp,a,c = process_image_bytes(b,min_contour_area=min_area)
            images_disp.append((disp,b))
            areas.append(a)
            counts.append(c)
        # align times
        while len(times)<len(areas): times.append(times[-1]+1.0)
        growth_rates=[]; phases=[]
        for i,a in enumerate(areas):
            if i==0:
                growth_rates.append(0.0); phases.append(detect_phase(0,a,1.0))
            else:
                dt = max(0.0001,times[i]-times[i-1])
                growth_rates.append((a-areas[i-1])/dt)
                phases.append(detect_phase(areas[i-1],a,dt))
        # show images
        st.subheader("Processed Images")
        cols=st.columns(len(images_disp))
        for i,(img_rgb,_) in enumerate(images_disp):
            cols[i].image(img_rgb, caption=f"t={times[i]}h | Area={areas[i]:.1f}px | Count={counts[i]} | Phase={phases[i]}", use_column_width=True)
        # df
        df=pd.DataFrame({"Day":list(range(1,len(areas)+1)),
                         "Time":times,
                         "Area":areas,
                         "Count":counts,
                         "GrowthRate_per_h":growth_rates,
                         "Phase":phases})
        st.subheader("Summary Table")
        st.dataframe(df)
        # smoothing
        if smoothing>0 and len(areas)>1:
            smooth_areas = list(gaussian_filter1d(areas,sigma=smoothing))
        else: smooth_areas=areas
        phase_color={"Lag":"blue","Log":"green","Stationary":"orange","Decline":"red","No colony":"gray"}
        colors=[phase_color.get(p,"black") for p in phases]
        # plot
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=times,y=smooth_areas,mode='lines+markers',name='Area (smoothed)',marker=dict(color='red')))
        for t,a,p in zip(times,areas,phases):
            fig.add_trace(go.Scatter(x=[t],y=[a],mode='markers',marker=dict(color=phase_color.get(p,"black"),size=10),name=f"{p}"))
        if show_pred and len(times)>=2:
            nxt_t,pred_area=predict_next_area(times,areas)
            fig.add_trace(go.Scatter(x=[nxt_t],y=[pred_area],mode='markers+text',marker=dict(symbol='diamond',size=12,color='purple'),text=["Pred"],textposition="top center",name="Prediction"))
            st.info(f"Predicted next: t={nxt_t:.2f}h → area={pred_area:.1f}px")
        fig.update_layout(title="Growth Curve (color-coded phases)", xaxis_title="Time (hours)", yaxis_title="Colony Area (pixels)")
        st.plotly_chart(fig,use_container_width=True)
        st.download_button("Download CSV", data=df_to_csv_bytes(df), file_name="growth_analyzer.csv", mime="text/csv")
         
        # PDF export
        if PDF_AVAILABLE and st.button("Download PDF Report (Normal Growth)"):
            pdf_bytes = make_pdf_report("Normal Growth Analyzer Report", df, None, [])
            if pdf_bytes is not None:
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name="normal_growth_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("PDF generation failed")
    else:
        st.warning("No data available to generate PDF")

# ---------- ANTIBIOTIC SIMULATION ----------
if mode== "Antibiotic Simulation":
    st.header("Antibiotic Simulation: Treated vs Untreated")
    st.write("Upload matched sets of images for untreated and treated groups (same timepoints).")

    col_a, col_b = st.columns(2)
    with col_a:
        u_files = st.file_uploader("Upload Untreated images", accept_multiple_files=True, type=["jpg","png","jpeg"], key="u")
    with col_b:
        t_files = st.file_uploader("Upload Treated images", accept_multiple_files=True, type=["jpg","png","jpeg"], key="t")

    # fallback sample generator
    if (not u_files) and (not t_files):
        if st.button("Generate demo untreated/treated sets"):
            radii = [20,50,90,130]
            u_files = []
            t_files = []
            for r in radii:
                img = np.ones((300,300,3), dtype=np.uint8)*255
                cv2.circle(img, (150,150), r, (0,0,0), -1)
                _, b = cv2.imencode(".png", img); u_files.append(BytesIO(b.tobytes()))
                img2 = np.ones((300,300,3), dtype=np.uint8)*255
                cv2.circle(img2, (150,150), int(max(5,r*0.5)), (0,0,0), -1)
                _, b2 = cv2.imencode(".png", img2); t_files.append(BytesIO(b2.tobytes()))
            st.experimental_rerun()

    # times input
    u_times, t_times = [], []
    if u_files:
        st.write("Times for Untreated images (hours):")
        for i, f in enumerate(u_files, start=1):
            val = st.number_input(f"Untreated time for image {i}", min_value=0.0, value=float(i-1), key=f"ut_{i}")
            u_times.append(float(val))
    if t_files:
        st.write("Times for Treated images (hours):")
        for i, f in enumerate(t_files, start=1):
            val = st.number_input(f"Treated time for image {i}", min_value=0.0, value=float(i-1), key=f"tt_{i}")
            t_times.append(float(val))

    min_area_ab = st.slider("Min contour area (px)", 10, 2000, 50)
    smoothing_ab = st.slider("Smoothing sigma (0=off)", 0.0, 3.0, 1.0)

    if u_files or t_files:
        # helper to process image list
        def process_list(files, times_list):
            areas=[]; counts=[]; disp_images=[]; raw_bytes=[]
            for i,f in enumerate(files):
                if hasattr(f,"read"):
                    f.seek(0); b=f.read()
                else:
                    b=f.getvalue()
                disp,a,cnt=process_image_bytes(b,min_contour_area=min_area_ab)
                disp_images.append(disp); raw_bytes.append(b)
                areas.append(a); counts.append(cnt)
            # align times
            if len(times_list) < len(areas):
                last = times_list[-1] if times_list else 0.0
                while len(times_list) < len(areas):
                    last += 1.0; times_list.append(last)
            # phases
            grs=[]; phs=[]
            for i,a in enumerate(areas):
                if i==0:
                    grs.append(0.0); phs.append(detect_phase(0,a,1.0))
                else:
                    dt = max(0.0001, times_list[i]-times_list[i-1])
                    grs.append((a-areas[i-1])/dt); phs.append(detect_phase(areas[i-1],a,dt))
            return {"times":times_list,"areas":areas,"counts":counts,"images":disp_images,"raw":raw_bytes,"growth":grs,"phases":phs}

        u_res = process_list(u_files, list(u_times)) if u_files else None
        t_res = process_list(t_files, list(t_times)) if t_files else None

        # display thumbnails side by side
        st.subheader("Side-by-side processed images (Untreated | Treated)")
        n = max(len(u_res["images"]) if u_res else 0, len(t_res["images"]) if t_res else 0)
        for i in range(n):
            cols = st.columns(2)
            if u_res and i < len(u_res["images"]):
                cols[0].image(u_res["images"][i], caption=f"Untreated t={u_res['times'][i]}h | Area={u_res['areas'][i]:.1f}")
            else:
                cols[0].write("No untreated")
            if t_res and i < len(t_res["images"]):
                cols[1].image(t_res["images"][i], caption=f"Treated t={t_res['times'][i]}h | Area={t_res['areas'][i]:.1f}")
            else:
                cols[1].write("No treated")

        # build dataframes
        df_u = pd.DataFrame({"Day": list(range(1,len(u_res["areas"])+1)), "Time": u_res["times"], "Area": u_res["areas"], "Phase": u_res["phases"]}) if u_res else pd.DataFrame()
        df_t = pd.DataFrame({"Day": list(range(1,len(t_res["areas"])+1)), "Time": t_res["times"], "Area": t_res["areas"], "Phase": t_res["phases"]}) if t_res else pd.DataFrame()

        # Comparison table
        st.subheader("Comparison Table")
        if not df_u.empty and not df_t.empty:
            m = min(len(df_u), len(df_t))
            comp = pd.DataFrame({
                "Day": list(range(1,m+1)),
                "Untreated_Area": df_u["Area"].tolist()[:m],
                "Treated_Area": df_t["Area"].tolist()[:m],
            })
            comp["Percent_Reduction"] = comp.apply(lambda r: 100.0*(r["Untreated_Area"]-r["Treated_Area"])/r["Untreated_Area"] if r["Untreated_Area"]>0 else 0.0, axis=1)
            st.table(comp)
        else:
            st.write("Provide both untreated and treated images to compare.")

        # --- Plotly figure with colors ---
        fig2 = go.Figure()

        # Untreated
        y_u = gaussian_filter1d(df_u["Area"].tolist(), sigma=smoothing_ab) if smoothing_ab>0 else df_u["Area"].tolist()
        fig2.add_trace(go.Scatter(
            x=df_u["Time"], y=y_u,
            mode='lines+markers',
            name='Untreated',
            line=dict(color='royalblue', width=3),
            marker=dict(color='lightblue', size=8)
        ))

        # Treated
        y_t = gaussian_filter1d(df_t["Area"].tolist(), sigma=smoothing_ab) if smoothing_ab>0 else df_t["Area"].tolist()
        fig2.add_trace(go.Scatter(
            x=df_t["Time"], y=y_t,
            mode='lines+markers',
            name='Treated',
            line=dict(color='crimson', width=3),
            marker=dict(color='pink', size=8)
        ))

        # % reduction annotation
        if not df_u.empty and not df_t.empty:
            last = min(len(df_u)-1, len(df_t)-1)
            u_last = df_u["Area"].iloc[last]; t_last = df_t["Area"].iloc[last]
            red = (u_last - t_last)/u_last*100 if u_last>0 else 0.0
            fig2.add_annotation(
                x=df_t["Time"].iloc[last],
                y=max(u_last,t_last),
                text=f"% reduction (last) = {red:.1f}%",
                showarrow=True
            )

        # Layout
        fig2.update_layout(
            title="📊 Treated vs Untreated Growth Curves",
            xaxis_title="Time (hours)",
            yaxis_title="Colony Area (px)",
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridcolor="lightgray"),
            font=dict(size=14)
        )

        st.plotly_chart(fig2, use_container_width=True)

        # --- Download CSV ---
        if not df_u.empty and not df_t.empty:
            st.download_button("Download comparison CSV", data=df_to_csv_bytes(comp), file_name="antibiotic_comparison.csv", mime="text/csv")

        # --- PDF report ---
        if PDF_AVAILABLE and st.button("Download PDF Report (Antibiotic Simulation)"):
            if not df_u.empty and not df_t.empty:
                # Add a column to distinguish groups
                df_pdf = pd.concat([
                    df_u.assign(Group="Untreated"),
                    df_t.assign(Group="Treated")
                ], ignore_index=True)
                pdf_bytes = make_pdf_report("Antibiotic Simulation Report", df_pdf, None, [])
                if pdf_bytes is not None:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name="antibiotic_simulation_report.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("PDF generation failed")
            else:
                st.warning("Provide both untreated and treated data to generate PDF")


# ---------- QR & Reports ----------
st.write("---")
st.header("📎 QR Code Generator")
link = st.text_input("Enter URL (app or repo)", value="")
if st.button("Generate QR Code"):
    if link:
        qr_img = qrcode.make(link)
        buf = BytesIO()
        qr_img.save(buf, format="PNG")
        st.image(buf.getvalue(), caption="Generated QR Code", use_column_width=False)
        st.download_button("Download QR Code", buf.getvalue(), file_name="QR_Code.png")
    else:
        st.warning("Please enter a valid link.")
