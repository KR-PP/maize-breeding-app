import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Maize Breeding Analysis",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 { font-size: 1.8rem; font-weight: 600; margin: 0; }
.main-header p  { font-size: 0.9rem; opacity: 0.85; margin: 0.3rem 0 0; }

.metric-card {
    background: white;
    border: 1px solid #e8f5e9;
    border-left: 4px solid #2e7d32;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .val { font-size: 1.8rem; font-weight: 600; color: #1b5e20; }
.metric-card .lbl { font-size: 0.75rem; color: #666; margin-top: 2px; }

.section-header {
    background: #f1f8e9;
    border-left: 4px solid #558b2f;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    font-weight: 600;
    color: #1b5e20;
    margin: 1.2rem 0 0.8rem;
    font-size: 1rem;
}
.tag-check  { background:#e3f2fd; color:#1565c0; padding:2px 8px; border-radius:12px; font-size:0.78rem; }
.tag-trial  { background:#e8f5e9; color:#2e7d32; padding:2px 8px; border-radius:12px; font-size:0.78rem; }
.tag-ws     { background:#fff3e0; color:#e65100; padding:2px 8px; border-radius:12px; font-size:0.78rem; }
.tag-ww     { background:#e1f5fe; color:#0277bd; padding:2px 8px; border-radius:12px; font-size:0.78rem; }
.stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ────────────────────────────────────────────────────────────
CHECKS = ['Pioneer', 'Pacific', 'Syngenta', 'DOA', 'NK', 'Pac', 'P4555', 'Pac789', 'NK6848', 'NS3', 'NS4', 'NS5']

def is_check(origin_str):
    if pd.isna(origin_str):
        return False
    return any(c.lower() in str(origin_str).lower() for c in CHECKS)

def load_nursery(f):
    df = pd.read_excel(f, sheet_name='ข้อมูลดิบ', header=None, skiprows=4)
    df.columns = range(len(df.columns))
    df = df[pd.to_numeric(df[0], errors='coerce').notna()].copy()
    col_map = {
        0: 'entry', 1: 'pedigree', 2: 'origin', 3: 'rows',
        4: 'tas_date', 5: 'silk_date', 6: 'plant_ht', 7: 'ear_ht',
        8: 'tas_sco', 9: 'plant_asp', 10: 'hk_cov', 11: 'ear_asp',
        12: 'n_ears', 13: 'weight_gm', 14: 'comments'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for c in ['entry', 'tas_date', 'silk_date', 'plant_ht', 'ear_ht', 'weight_gm']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def load_yield_trial(f, trial_type='WW'):
    df = pd.read_excel(f, sheet_name='ข้อมูลดิบ', header=None, skiprows=4)
    df.columns = range(len(df.columns))
    df = df[pd.to_numeric(df[0], errors='coerce').notna()].copy()
    ncols = df.shape[1]

    if ncols >= 34:  # WS format (34 cols)
        col_map = {
            0:'rep', 1:'blk', 2:'entry', 3:'plot',
            4:'pedigree', 5:'origin', 6:'rows',
            7:'days_tass', 8:'days_silk',
            9:'plant_ht', 10:'ear_ht',
            11:'root_lodge', 12:'stalk_lodge',
            13:'plant_count', 14:'stand_count',
            15:'n_ears_total', 16:'ear_rot',
            17:'field_wt', 18:'grain_wt',
            19:'moist_pct', 20:'shell_pct',
            21:'plant_asp', 22:'ear_asp',
            23:'open_hk', 24:'tip_fill',
            25:'seed_vigor',
            32:'yield_kg_rai', 33:'grain_type'
        }
    elif ncols >= 29:  # WW format (29 cols)
        col_map = {
            0:'rep', 1:'blk', 2:'entry', 3:'plot',
            4:'pedigree', 5:'origin', 6:'rows',
            7:'days_tass', 8:'days_silk',
            9:'plant_ht', 10:'ear_ht',
            11:'root_lodge', 12:'stalk_lodge',
            13:'plant_count', 14:'stand_count',
            15:'n_ears_total', 16:'ear_rot',
            17:'field_wt', 18:'grain_wt',
            19:'moist_pct', 20:'shell_pct',
            21:'plant_asp', 22:'ear_asp',
            23:'open_hk', 24:'tip_fill',
            25:'seed_vigor',
            26:'yield_kg_rai', 27:'grain_type'
        }
    else:  # YT226102 format (no pedigree col)
        col_map = {
            0:'rep', 1:'blk', 2:'entry', 3:'plot',
            4:'days_tass', 5:'days_silk',
            6:'plant_ht', 7:'ear_ht',
            8:'root_lodge', 9:'stalk_lodge',
            10:'plant_count', 11:'stand_count',
            12:'n_ears_total', 13:'ear_rot',
            14:'field_wt', 15:'grain_wt',
            16:'shell_pct', 17:'moist_pct',
            18:'plant_asp', 19:'ear_asp',
            20:'open_hk', 21:'tip_fill',
            22:'seed_vigor',
            23:'yield_kg_rai', 24:'grain_type'
        }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    num_cols = ['rep','blk','entry','plot','days_tass','days_silk',
                'plant_ht','ear_ht','plant_count','stand_count',
                'n_ears_total','field_wt','grain_wt','moist_pct',
                'shell_pct','yield_kg_rai','root_lodge','stalk_lodge']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'origin' in df.columns:
        df['is_check'] = df['origin'].apply(is_check)
    else:
        df['is_check'] = False

    df['trial_type'] = trial_type
    df['yield_ton_rai'] = df['yield_kg_rai'] / 1000 if 'yield_kg_rai' in df.columns else np.nan
    return df

def load_pedigree_from_source(f):
    try:
        xl = pd.ExcelFile(f)
        if 'source' in xl.sheet_names:
            df = pd.read_excel(f, sheet_name='source', header=None, skiprows=1)
            df.columns = range(len(df.columns))
            df = df.rename(columns={0:'plot', 1:'block', 2:'trt', 3:'entry', 4:'pedigree', 5:'origin'})
            return df[['entry','pedigree','origin']].dropna(subset=['entry'])
    except:
        pass
    return pd.DataFrame()

# ─── STATISTICS ──────────────────────────────────────────────────────────────
def calc_means_vs_checks(df, trait='yield_ton_rai'):
    if trait not in df.columns:
        return pd.DataFrame()
    checks = df[df['is_check'] == True]
    trials = df[df['is_check'] == False]
    check_mean = checks[trait].mean() if len(checks) > 0 else np.nan
    grand_mean = df[trait].mean()

    result = trials.groupby('entry')[trait].agg(['mean','std','count']).reset_index()
    result.columns = ['entry', 'mean', 'std', 'n_reps']

    if 'pedigree' in df.columns:
        ped = df[['entry','pedigree','origin']].drop_duplicates('entry')
        result = result.merge(ped, on='entry', how='left')

    result['pct_of_check'] = (result['mean'] / check_mean * 100).round(1) if not np.isnan(check_mean) else np.nan
    result['pct_of_mean']  = (result['mean'] / grand_mean * 100).round(1)
    result['rank'] = result['mean'].rank(ascending=False).astype(int)
    result = result.sort_values('rank')
    result['check_mean'] = round(check_mean, 3) if not np.isnan(check_mean) else np.nan
    return result

def calc_drought_index(ws_df, ww_df):
    ws_means = ws_df[ws_df['is_check']==False].groupby('entry')['yield_ton_rai'].mean()
    ww_means = ww_df[ww_df['is_check']==False].groupby('entry')['yield_ton_rai'].mean()

    ws_check = ws_df[ws_df['is_check']==True]['yield_ton_rai'].mean()
    ww_check = ww_df[ww_df['is_check']==True]['yield_ton_rai'].mean()

    combined = pd.DataFrame({'yield_ws': ws_means, 'yield_ww': ww_means}).dropna()
    combined['DTI'] = (combined['yield_ws'] * combined['yield_ww']) / (ws_check * ww_check)
    combined['SSI'] = (1 - combined['yield_ws']/combined['yield_ww']) / (1 - ws_check/ww_check)
    combined['stress_reduction_pct'] = ((combined['yield_ww'] - combined['yield_ws']) / combined['yield_ww'] * 100).round(1)
    combined['DTI'] = combined['DTI'].round(4)
    combined['SSI'] = combined['SSI'].round(4)
    combined = combined.reset_index()

    if 'pedigree' in ws_df.columns:
        ped = ws_df[['entry','pedigree','origin']].drop_duplicates('entry')
        combined = combined.merge(ped, on='entry', how='left')

    combined['drought_class'] = combined.apply(lambda r:
        '🏆 ทนแล้งดีเยี่ยม' if r['DTI'] >= 0.8 and r['SSI'] <= 0.8 else
        ('✅ ทนแล้งดี'      if r['DTI'] >= 0.6 and r['SSI'] <= 1.0 else
        ('⚠️ ปานกลาง'      if r['DTI'] >= 0.4 else '❌ อ่อนแอ')), axis=1)
    return combined.sort_values('DTI', ascending=False)

def calc_augmented_rcbd(df, trait='yield_ton_rai'):
    """Adjusted means using check correction for Augmented RCBD"""
    if trait not in df.columns or 'blk' not in df.columns:
        return df
    checks = df[df['is_check']==True]
    if len(checks) == 0:
        return df
    blk_check_mean = checks.groupby('blk')[trait].mean()
    overall_check_mean = checks[trait].mean()
    blk_effect = blk_check_mean - overall_check_mean

    df = df.copy()
    df['blk_effect'] = df['blk'].map(blk_effect).fillna(0)
    df['yield_adj'] = df[trait] - df['blk_effect']
    return df

def cv_percent(series):
    m = series.mean()
    if m == 0 or np.isnan(m):
        return np.nan
    return round(series.std() / m * 100, 1)

# ─── UI HELPERS ──────────────────────────────────────────────────────────────
def metric_row(metrics):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{value}</div>
                <div class="lbl">{label}</div>
            </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌽 Maize Breeding")
    st.markdown("**NSFCRC Breeding Program**")
    st.divider()

    page = st.radio("📋 เลือก Module", [
        "🏠 หน้าหลัก",
        "🌿 Nursery Analysis",
        "📊 Yield Trial Analysis",
        "💧 Drought Tolerance (WS vs WW)",
        "📈 Multi-Trial Summary"
    ])

    st.divider()
    st.markdown("**📁 Upload ไฟล์**")
    nursery_files = st.file_uploader("Nursery (P……xlsx)", type=['xlsx'], accept_multiple_files=True)
    ws_files = st.file_uploader("Yield Trial WS (……WS.xlsx)", type=['xlsx'], accept_multiple_files=True)
    ww_files = st.file_uploader("Yield Trial WW (……WW.xlsx)", type=['xlsx'], accept_multiple_files=True)
    yt_files = st.file_uploader("Yield Trial อื่นๆ (YT……xlsx)", type=['xlsx'], accept_multiple_files=True)

    st.divider()
    st.caption("v1.0 | NSFCRC 2026")

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
@st.cache_data
def cached_load_nursery(file_bytes, name):
    import io
    return load_nursery(io.BytesIO(file_bytes))

@st.cache_data
def cached_load_yt(file_bytes, name, ttype):
    import io
    return load_yield_trial(io.BytesIO(file_bytes), ttype)

nursery_data = {}
for f in nursery_files:
    try:
        nursery_data[f.name] = cached_load_nursery(f.read(), f.name)
    except Exception as e:
        st.sidebar.error(f"❌ {f.name}: {e}")

ws_data = {}
for f in ws_files:
    try:
        ws_data[f.name] = cached_load_yt(f.read(), f.name, 'WS')
    except Exception as e:
        st.sidebar.error(f"❌ {f.name}: {e}")

ww_data = {}
for f in ww_files:
    try:
        ww_data[f.name] = cached_load_yt(f.read(), f.name, 'WW')
    except Exception as e:
        st.sidebar.error(f"❌ {f.name}: {e}")

yt_data = {}
for f in yt_files:
    try:
        yt_data[f.name] = cached_load_yt(f.read(), f.name, 'YT')
    except Exception as e:
        st.sidebar.error(f"❌ {f.name}: {e}")

all_yt = {**ws_data, **ww_data, **yt_data}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 หน้าหลัก":
    st.markdown("""
    <div class="main-header">
        <h1>🌽 Maize Breeding Analysis System</h1>
        <p>NSFCRC · Drought Tolerance Breeding Program · 2026 D</p>
    </div>
    """, unsafe_allow_html=True)

    total_entries = sum(len(d) for d in nursery_data.values())
    total_yt_obs  = sum(len(d) for d in all_yt.values())
    n_trials      = len(all_yt)
    n_nursery     = len(nursery_data)

    metric_row([
        ("Nursery Files", n_nursery),
        ("Nursery Entries", total_entries),
        ("Yield Trial Files", n_trials),
        ("YT Observations", total_yt_obs),
    ])

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        section("📂 ไฟล์ที่ Upload แล้ว")
        if nursery_data:
            for n, df in nursery_data.items():
                st.success(f"🌿 **{n}** — {len(df)} entries")
        if ws_data:
            for n, df in ws_data.items():
                checks = df['is_check'].sum()
                trials = (~df['is_check']).sum()
                st.warning(f"💧 **{n}** (WS) — {trials} entries, {checks} check obs")
        if ww_data:
            for n, df in ww_data.items():
                checks = df['is_check'].sum()
                trials = (~df['is_check']).sum()
                st.info(f"💦 **{n}** (WW) — {trials} entries, {checks} check obs")
        if yt_data:
            for n, df in yt_data.items():
                st.success(f"📊 **{n}** — {len(df)} obs")
        if not any([nursery_data, ws_data, ww_data, yt_data]):
            st.info("👈 Upload ไฟล์ที่ Sidebar ทางซ้ายก่อนครับ")

    with col2:
        section("📋 วิธีใช้งาน")
        st.markdown("""
        1. **Upload ไฟล์** ที่ Sidebar ซ้าย
           - Nursery = ไฟล์ P……xlsx
           - Yield WS = ไฟล์ที่มี WS ในชื่อ
           - Yield WW = ไฟล์ที่มี WW ในชื่อ
           - Yield อื่นๆ = YT…… ที่ไม่ใช่ WS/WW
        2. **เลือก Module** ที่ต้องการวิเคราะห์
        3. **ดูผลและ Download** รายงาน

        ---
        **ไฟล์ที่รองรับ:** format ข้อมูลดิบ NSFCRC
        """)

        section("🔬 Modules ที่มี")
        st.markdown("""
        - 🌿 **Nursery** — Selfing selection, Pedigree tracking
        - 📊 **Yield Trial** — Adjusted means vs checks, Ranking
        - 💧 **Drought Tolerance** — DTI, SSI, WS vs WW comparison
        - 📈 **Multi-Trial** — Summary across trials
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: NURSERY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌿 Nursery Analysis":
    st.markdown('<div class="main-header"><h1>🌿 Nursery Analysis</h1><p>Line Development · Selfing Selection</p></div>', unsafe_allow_html=True)

    if not nursery_data:
        st.info("👈 Upload ไฟล์ Nursery (P……xlsx) ก่อนครับ")
        st.stop()

    selected = st.selectbox("เลือกไฟล์ Nursery", list(nursery_data.keys()))
    df = nursery_data[selected].copy()

    # Summary metrics
    has_weight = df['weight_gm'].notna().sum()
    has_silk   = df['silk_date'].notna().sum()
    avg_silk   = df['silk_date'].mean()

    metric_row([
        ("Total Entries", len(df)),
        ("มีข้อมูลน้ำหนัก", has_weight),
        ("มีข้อมูล Silk Date", has_silk),
        ("Avg Silk Date", f"{avg_silk:.0f}" if not np.isnan(avg_silk) else "-"),
    ])

    tab1, tab2, tab3 = st.tabs(["📋 ข้อมูลทั้งหมด", "📊 กราฟ", "🏆 Selection"])

    with tab1:
        section("ข้อมูล Nursery ทั้งหมด")
        show_cols = [c for c in ['entry','pedigree','origin','tas_date','silk_date',
                                  'plant_ht','ear_ht','plant_asp','ear_asp',
                                  'n_ears','weight_gm','comments'] if c in df.columns]
        st.dataframe(df[show_cols].reset_index(drop=True), use_container_width=True, height=500)

        csv = df[show_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ Download CSV", csv, f"{selected}_nursery.csv", "text/csv")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            section("น้ำหนักฝัก (gm) Distribution")
            if df['weight_gm'].notna().sum() > 0:
                fig = px.histogram(df.dropna(subset=['weight_gm']), x='weight_gm',
                                   nbins=20, color_discrete_sequence=['#388e3c'],
                                   labels={'weight_gm': 'น้ำหนักฝัก (gm)'})
                fig.update_layout(showlegend=False, height=350,
                                  plot_bgcolor='white', paper_bgcolor='white')
                fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ยังไม่มีข้อมูลน้ำหนักฝัก")

        with col2:
            section("Silk Date Distribution")
            if df['silk_date'].notna().sum() > 0:
                fig = px.histogram(df.dropna(subset=['silk_date']), x='silk_date',
                                   nbins=15, color_discrete_sequence=['#1565c0'],
                                   labels={'silk_date': 'วันออกไหม (DAP)'})
                fig.update_layout(showlegend=False, height=350,
                                  plot_bgcolor='white', paper_bgcolor='white')
                fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
                fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ยังไม่มีข้อมูล Silk Date")

        if df['weight_gm'].notna().sum() > 0 and df['silk_date'].notna().sum() > 0:
            section("น้ำหนักฝัก vs Silk Date")
            fig = px.scatter(df.dropna(subset=['weight_gm','silk_date']),
                             x='silk_date', y='weight_gm',
                             hover_data=['entry','pedigree'] if 'pedigree' in df.columns else ['entry'],
                             color_discrete_sequence=['#2e7d32'],
                             labels={'silk_date':'Silk Date (DAP)', 'weight_gm':'น้ำหนักฝัก (gm)'})
            fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("🏆 คัดเลือกสายพันธุ์ที่ดีที่สุด")

        col1, col2 = st.columns(2)
        with col1:
            sel_pct = st.slider("Selection intensity (%)", 5, 50, 20, 5)
        with col2:
            sel_trait = st.selectbox("Trait สำหรับ selection",
                ['weight_gm','silk_date','plant_ht','ear_ht'],
                format_func=lambda x: {'weight_gm':'น้ำหนักฝัก','silk_date':'Silk Date',
                                        'plant_ht':'ความสูงต้น','ear_ht':'ความสูงฝัก'}.get(x, x))

        if sel_trait in df.columns and df[sel_trait].notna().sum() > 0:
            trait_df = df.dropna(subset=[sel_trait]).copy()
            n_select = max(1, int(len(trait_df) * sel_pct / 100))
            ascending = sel_trait == 'silk_date'
            selected_df = trait_df.nsmallest(n_select, sel_trait) if ascending else trait_df.nlargest(n_select, sel_trait)

            st.success(f"✅ คัดเลือก **{n_select}** สายพันธุ์ จาก {len(trait_df)} (top {sel_pct}%)")

            show_sel = [c for c in ['entry','pedigree','origin', sel_trait,'plant_asp','ear_asp','comments'] if c in selected_df.columns]
            st.dataframe(selected_df[show_sel].reset_index(drop=True),
                         use_container_width=True, height=400)

            csv2 = selected_df[show_sel].to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ Download Selected Entries", csv2,
                               f"selected_{sel_pct}pct_{sel_trait}.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: YIELD TRIAL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Yield Trial Analysis":
    st.markdown('<div class="main-header"><h1>📊 Yield Trial Analysis</h1><p>Adjusted Means · Ranking vs Checks · Augmented RCBD</p></div>', unsafe_allow_html=True)

    if not all_yt:
        st.info("👈 Upload ไฟล์ Yield Trial ก่อนครับ")
        st.stop()

    selected = st.selectbox("เลือก Trial", list(all_yt.keys()))
    df_raw = all_yt[selected].copy()

    # Augmented RCBD adjustment
    df = calc_augmented_rcbd(df_raw, 'yield_ton_rai')
    use_adj = 'yield_adj' in df.columns and df['yield_adj'].notna().sum() > 0
    yield_col = 'yield_adj' if use_adj else 'yield_ton_rai'

    n_reps    = df['rep'].nunique() if 'rep' in df.columns else 0
    n_entries = df[df['is_check']==False]['entry'].nunique()
    n_checks  = df[df['is_check']==True]['origin'].nunique() if 'origin' in df.columns else 0
    check_mean = df[df['is_check']==True]['yield_ton_rai'].mean()
    grand_mean = df['yield_ton_rai'].mean()
    cv = cv_percent(df['yield_ton_rai'])

    metric_row([
        ("Entries (ไม่รวม check)", n_entries),
        ("Checks", n_checks),
        ("Reps", n_reps),
        ("Check Mean (ตัน/ไร่)", f"{check_mean:.3f}" if not np.isnan(check_mean) else "-"),
        ("Grand Mean", f"{grand_mean:.3f}" if not np.isnan(grand_mean) else "-"),
        ("CV (%)", f"{cv}" if cv else "-"),
    ])

    if use_adj:
        st.success("✅ ใช้ **Augmented RCBD Adjusted Means** (ปรับ block effect ด้วย checks แล้ว)")

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking", "📊 กราฟ", "🔍 Data ทั้งหมด", "📋 Checks"])

    with tab1:
        section("🏆 Ranking — Adjusted Mean Yield")

        result = calc_means_vs_checks(df, yield_col)
        if len(result) == 0:
            st.warning("ไม่มีข้อมูลเพียงพอ")
        else:
            # Filter controls
            col1, col2, col3 = st.columns(3)
            with col1:
                top_n = st.slider("แสดง Top N", 10, len(result), min(50, len(result)), 5)
            with col2:
                min_pct = st.number_input("% ของ Check ขั้นต่ำ", 0, 200, 0)
            with col3:
                st.metric("Check Mean", f"{check_mean:.3f} ตัน/ไร่" if not np.isnan(check_mean) else "-")

            filtered = result[result['pct_of_check'] >= min_pct].head(top_n) if 'pct_of_check' in result.columns else result.head(top_n)

            display_cols = [c for c in ['rank','entry','pedigree','origin','mean','pct_of_check','n_reps'] if c in filtered.columns]
            rename_map = {'rank':'อันดับ','entry':'Entry','pedigree':'Pedigree','origin':'Origin',
                          'mean':'Mean Yield (ตัน/ไร่)','pct_of_check':'% of Check','n_reps':'N Reps'}

            disp = filtered[display_cols].rename(columns=rename_map)
            disp['Mean Yield (ตัน/ไร่)'] = disp['Mean Yield (ตัน/ไร่)'].round(3)

            st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=500)

            csv = filtered.to_csv(index=False).encode('utf-8-sig')
            st.download_button("⬇️ Download Ranking", csv, f"ranking_{selected}.csv", "text/csv")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            section("Yield Distribution")
            result2 = calc_means_vs_checks(df, yield_col)
            if len(result2) > 0:
                fig = px.histogram(result2, x='mean', nbins=25,
                                   color_discrete_sequence=['#388e3c'],
                                   labels={'mean': 'Mean Yield (ตัน/ไร่)', 'count': 'จำนวน Entries'})
                if not np.isnan(check_mean):
                    fig.add_vline(x=check_mean, line_dash='dash', line_color='red',
                                  annotation_text=f"Check mean: {check_mean:.2f}")
                fig.update_layout(height=380, plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            section("Top 20 Entries vs Checks")
            result3 = calc_means_vs_checks(df, yield_col)
            if len(result3) > 0:
                top20 = result3.head(20).copy()
                checks_plot = df[df['is_check']==True].copy()
                if len(checks_plot) > 0 and 'origin' in checks_plot.columns:
                    ck_means = checks_plot.groupby('origin')[yield_col if yield_col in checks_plot else 'yield_ton_rai'].mean().reset_index()
                    ck_means.columns = ['label','mean']
                    ck_means['type'] = 'Check'
                    top20['label'] = top20['entry'].astype(str)
                    top20['type'] = 'Entry'
                    plot_df = pd.concat([
                        top20[['label','mean','type']],
                        ck_means[['label','mean','type']]
                    ])
                else:
                    top20['label'] = top20['entry'].astype(str)
                    top20['type'] = 'Entry'
                    plot_df = top20[['label','mean','type']]

                color_map = {'Entry': '#2e7d32', 'Check': '#e53935'}
                fig = px.bar(plot_df, x='label', y='mean', color='type',
                             color_discrete_map=color_map,
                             labels={'mean': 'Yield (ตัน/ไร่)', 'label': ''})
                fig.update_layout(height=380, plot_bgcolor='white',
                                  paper_bgcolor='white', showlegend=True)
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        # Trait correlation
        section("Trait Correlations")
        num_traits = [c for c in ['yield_ton_rai','days_silk','plant_ht','ear_ht',
                                   'moist_pct','shell_pct','plant_asp','ear_asp']
                      if c in df.columns and df[c].notna().sum() > 10]

        if len(num_traits) >= 3:
            corr_df = df[df['is_check']==False][num_traits].corr().round(2)
            fig = px.imshow(corr_df, text_auto=True, aspect='auto',
                            color_continuous_scale='RdYlGn',
                            zmin=-1, zmax=1)
            fig.update_layout(height=420, title="Pearson Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("ข้อมูลดิบทั้งหมด")
        show_cols = [c for c in ['rep','blk','entry','plot','pedigree','origin',
                                  'days_tass','days_silk','plant_ht','ear_ht',
                                  'yield_ton_rai','yield_adj','moist_pct','shell_pct',
                                  'plant_asp','ear_asp','is_check'] if c in df.columns]
        st.dataframe(df[show_cols].reset_index(drop=True),
                     use_container_width=True, height=500)
        csv = df[show_cols].to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ Download Raw Data", csv, f"rawdata_{selected}.csv", "text/csv")

    with tab4:
        section("📋 Check Variety Performance")
        checks_df = df[df['is_check']==True]
        if len(checks_df) > 0 and 'origin' in checks_df.columns:
            ck_summary = checks_df.groupby('origin').agg(
                mean_yield=('yield_ton_rai','mean'),
                n_obs=('yield_ton_rai','count'),
                std=('yield_ton_rai','std')
            ).reset_index()
            ck_summary['mean_yield'] = ck_summary['mean_yield'].round(3)
            ck_summary['std'] = ck_summary['std'].round(3)
            ck_summary = ck_summary.sort_values('mean_yield', ascending=False)
            st.dataframe(ck_summary.reset_index(drop=True), use_container_width=True)
        else:
            st.info("ไม่พบข้อมูล Check ในไฟล์นี้")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DROUGHT TOLERANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💧 Drought Tolerance (WS vs WW)":
    st.markdown('<div class="main-header"><h1>💧 Drought Tolerance Analysis</h1><p>Water Stress vs Well Water · DTI · SSI · Stress Susceptibility</p></div>', unsafe_allow_html=True)

    if not ws_data or not ww_data:
        st.info("👈 Upload ไฟล์ Yield Trial WS **และ** WW ก่อนครับ")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        ws_sel = st.selectbox("ไฟล์ Water Stress (WS)", list(ws_data.keys()))
    with col2:
        ww_sel = st.selectbox("ไฟล์ Well Water (WW)", list(ww_data.keys()))

    ws_df = ws_data[ws_sel]
    ww_df = ww_data[ww_sel]

    ws_mean = ws_df[ws_df['is_check']==False]['yield_ton_rai'].mean()
    ww_mean = ww_df[ww_df['is_check']==False]['yield_ton_rai'].mean()
    ws_check = ws_df[ws_df['is_check']==True]['yield_ton_rai'].mean()
    ww_check = ww_df[ww_df['is_check']==True]['yield_ton_rai'].mean()
    stress_red = (ww_mean - ws_mean) / ww_mean * 100 if ww_mean > 0 else 0

    metric_row([
        ("Mean Yield WW (ตัน/ไร่)", f"{ww_mean:.3f}"),
        ("Mean Yield WS (ตัน/ไร่)", f"{ws_mean:.3f}"),
        ("Stress Reduction (%)", f"{stress_red:.1f}%"),
        ("Check WW (ตัน/ไร่)", f"{ww_check:.3f}" if not np.isnan(ww_check) else "-"),
        ("Check WS (ตัน/ไร่)", f"{ws_check:.3f}" if not np.isnan(ws_check) else "-"),
    ])

    dti_df = calc_drought_index(ws_df, ww_df)

    if len(dti_df) == 0:
        st.warning("ไม่สามารถจับคู่ข้อมูลได้ — ตรวจสอบว่า Entry No. ตรงกันทั้งสองไฟล์")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🏆 DTI Ranking", "📊 กราฟ", "📋 ตารางสรุป"])

    with tab1:
        section("🏆 Drought Tolerance Index (DTI) Ranking")
        st.markdown("""
        - **DTI ≥ 0.8 + SSI ≤ 0.8** → 🏆 ทนแล้งดีเยี่ยม
        - **DTI ≥ 0.6 + SSI ≤ 1.0** → ✅ ทนแล้งดี
        - **DTI ≥ 0.4** → ⚠️ ปานกลาง
        - **DTI < 0.4** → ❌ อ่อนแอ
        """)

        show_cols = [c for c in ['entry','pedigree','origin','yield_ww','yield_ws',
                                  'stress_reduction_pct','DTI','SSI','drought_class']
                     if c in dti_df.columns]
        rename_map = {
            'entry':'Entry', 'pedigree':'Pedigree', 'origin':'Origin',
            'yield_ww':'Yield WW', 'yield_ws':'Yield WS',
            'stress_reduction_pct':'Stress Reduction (%)',
            'DTI':'DTI', 'SSI':'SSI', 'drought_class':'การจำแนก'
        }
        disp = dti_df[show_cols].rename(columns=rename_map).reset_index(drop=True)
        for c in ['Yield WW','Yield WS']:
            if c in disp:
                disp[c] = disp[c].round(3)

        st.dataframe(disp, use_container_width=True, height=500)

        csv = dti_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇️ Download DTI Results", csv, "drought_tolerance_results.csv", "text/csv")

        # Count by class
        section("สรุปจำนวนตามระดับความทนแล้ง")
        if 'drought_class' in dti_df.columns:
            summary = dti_df['drought_class'].value_counts().reset_index()
            summary.columns = ['ระดับ', 'จำนวน']
            cols = st.columns(len(summary))
            for col, (_, row) in zip(cols, summary.iterrows()):
                col.metric(row['ระดับ'], row['จำนวน'])

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            section("WS vs WW Yield Scatter")
            if 'drought_class' in dti_df.columns:
                color_map = {
                    '🏆 ทนแล้งดีเยี่ยม': '#1b5e20',
                    '✅ ทนแล้งดี': '#388e3c',
                    '⚠️ ปานกลาง': '#f57c00',
                    '❌ อ่อนแอ': '#c62828'
                }
                hover_cols = [c for c in ['entry','pedigree','DTI','SSI'] if c in dti_df.columns]
                fig = px.scatter(dti_df, x='yield_ww', y='yield_ws',
                                 color='drought_class',
                                 color_discrete_map=color_map,
                                 hover_data=hover_cols,
                                 labels={'yield_ww':'Yield WW (ตัน/ไร่)',
                                         'yield_ws':'Yield WS (ตัน/ไร่)',
                                         'drought_class':'การจำแนก'})
                # 1:1 line
                max_val = max(dti_df['yield_ww'].max(), dti_df['yield_ws'].max())
                fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                         mode='lines', line=dict(dash='dash', color='gray'),
                                         name='1:1 line', showlegend=True))
                fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            section("DTI Distribution")
            fig = px.histogram(dti_df, x='DTI', nbins=20,
                               color_discrete_sequence=['#1565c0'],
                               labels={'DTI': 'Drought Tolerance Index (DTI)'})
            fig.add_vline(x=dti_df['DTI'].mean(), line_dash='dash', line_color='red',
                          annotation_text=f"Mean: {dti_df['DTI'].mean():.3f}")
            fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

        section("DTI vs SSI (Drought Classification Plot)")
        fig = px.scatter(dti_df, x='SSI', y='DTI',
                         color='drought_class' if 'drought_class' in dti_df.columns else None,
                         hover_data=[c for c in ['entry','pedigree','yield_ws','yield_ww'] if c in dti_df.columns],
                         labels={'SSI':'Stress Susceptibility Index (SSI)',
                                 'DTI':'Drought Tolerance Index (DTI)'})
        fig.add_hline(y=dti_df['DTI'].mean(), line_dash='dot', line_color='green',
                      annotation_text="Mean DTI")
        fig.add_vline(x=1.0, line_dash='dot', line_color='orange',
                      annotation_text="SSI=1.0")
        fig.update_layout(height=450, plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        section("ตารางสรุปทั้งหมด")
        st.dataframe(dti_df.round(3).reset_index(drop=True), use_container_width=True, height=500)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MULTI-TRIAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Multi-Trial Summary":
    st.markdown('<div class="main-header"><h1>📈 Multi-Trial Summary</h1><p>Overview across all uploaded trials</p></div>', unsafe_allow_html=True)

    if not all_yt and not nursery_data:
        st.info("👈 Upload ไฟล์ก่อนครับ")
        st.stop()

    section("📊 สรุปทุก Trial")
    rows = []
    for name, df in all_yt.items():
        ttype = df['trial_type'].iloc[0] if 'trial_type' in df.columns else '-'
        n_entries = df[df['is_check']==False]['entry'].nunique()
        n_checks = df[df['is_check']==True]['origin'].nunique() if 'origin' in df.columns else 0
        mean_yield = df['yield_ton_rai'].mean()
        check_yield = df[df['is_check']==True]['yield_ton_rai'].mean()
        cv = cv_percent(df['yield_ton_rai'])
        rows.append({
            'Trial': name, 'Type': ttype,
            'Entries': n_entries, 'Checks': n_checks,
            'Grand Mean (ตัน/ไร่)': round(mean_yield, 3),
            'Check Mean (ตัน/ไร่)': round(check_yield, 3) if not np.isnan(check_yield) else '-',
            'CV (%)': cv
        })

    if rows:
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True)

        section("Grand Mean ของแต่ละ Trial")
        fig = px.bar(summary_df, x='Trial', y='Grand Mean (ตัน/ไร่)',
                     color='Type', barmode='group',
                     color_discrete_sequence=['#e65100', '#0277bd', '#2e7d32'],
                     labels={'Grand Mean (ตัน/ไร่)': 'Grand Mean (ตัน/ไร่)'})
        fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white',
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    if nursery_data:
        section("📋 Nursery Summary")
        n_rows = []
        for name, df in nursery_data.items():
            n_rows.append({
                'Nursery': name,
                'Entries': len(df),
                'มีน้ำหนัก': df['weight_gm'].notna().sum(),
                'Avg Weight (gm)': round(df['weight_gm'].mean(), 1) if df['weight_gm'].notna().sum() > 0 else '-',
                'Avg Silk Date': round(df['silk_date'].mean(), 1) if df['silk_date'].notna().sum() > 0 else '-',
            })
        st.dataframe(pd.DataFrame(n_rows), use_container_width=True)
