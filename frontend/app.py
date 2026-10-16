import random
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Smart Campus Safety", page_icon="🛡️", layout="wide")

NAVY, BLUE = "#0B1F3A", "#1E5AA8"
SEV_COLOR = {"High": "#D64545", "Medium": "#F0A202", "Low": "#2E9E6B"}
MOD_COLORS = ["#1E5AA8", "#F0A202", "#D64545", "#7B5EA7"]

# ---------------- Styling ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1F3A 0%, #12335F 100%); }
[data-testid="stSidebar"] * { color: #E8EEF7 !important; }
[data-testid="stSidebar"] input { color: #0B1F3A !important; }

.banner { background: linear-gradient(120deg, #0B1F3A, #1E5AA8); color: white; padding: 22px 28px;
          border-radius: 14px; margin-bottom: 22px; box-shadow: 0 6px 18px rgba(11,31,58,.25); }
.banner h1 { margin: 0; font-size: 26px; font-weight: 700; color: white; }
.banner p { margin: 4px 0 0 0; opacity: .8; font-size: 14px; }

.kpi { background: white; border-radius: 12px; padding: 16px 20px; box-shadow: 0 2px 10px rgba(11,31,58,.08);
       border-left: 6px solid var(--c); }
.kpi .label { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: #6B7A90; font-weight: 600; }
.kpi .value { font-size: 32px; font-weight: 700; color: #0B1F3A; line-height: 1.2; }

.section { font-size: 18px; font-weight: 600; color: #0B1F3A; margin: 26px 0 12px 0;
           padding-bottom: 6px; border-bottom: 2px solid #E1E8F2; }

.cam { background: white; border-radius: 12px; padding: 12px; box-shadow: 0 2px 10px rgba(11,31,58,.08); margin-bottom: 14px; }
.cam .name { font-weight: 600; color: #0B1F3A; margin-bottom: 8px; display: flex; justify-content: space-between; }
.pill { padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; color: white; }

.alert { background: white; border-radius: 10px; padding: 12px 18px; margin-bottom: 10px; display: flex;
         justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(11,31,58,.07);
         border-left: 5px solid var(--c); }
.alert .t { font-weight: 600; color: #0B1F3A; }
.alert .m { font-size: 12px; color: #6B7A90; }
.alert .r { text-align: right; }

.stButton > button, .stDownloadButton > button { background: #1E5AA8; color: white; border: 0;
        border-radius: 8px; padding: 8px 20px; font-weight: 600; }
.stButton > button:hover { background: #0B1F3A; color: white; }
</style>
""", unsafe_allow_html=True)


def banner(title, subtitle):
    st.markdown(f'<div class="banner"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(label, value, color=BLUE):
    st.markdown(f'<div class="kpi" style="--c:{color}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>', unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="section">{text}</div>', unsafe_allow_html=True)


def sev_style(v):
    return f"background-color:{SEV_COLOR.get(v, '#fff')}22; color:{SEV_COLOR.get(v, '#000')}; font-weight:600"


def style_layout(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_family="Inter", margin=dict(t=50, l=10, r=10, b=10), title_font_size=16)
    return fig


# ---------------- Sidebar ----------------
st.sidebar.markdown("## 🛡️ Smart Campus\n**Safety & Surveillance**")
st.sidebar.markdown("---")
PAGES = ["📹  Live Feeds", "🚨  Alerts & Events", "🚗  Vehicle / ANPR Log",
         "📊  Metrics & Reports", "🔔  Notifications"]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
BACKEND_URL = st.sidebar.text_input("Backend URL (FastAPI)", "", help="Empty = demo data")
REFRESH = st.sidebar.slider("Auto-refresh (sec)", 5, 60, 10)
st.sidebar.caption("Security Control Room • v1.0")

# ---------------- Data layer ----------------
MODULES = ["Vehicle + ANPR", "Helmet + Triple Riding", "Threat Detection", "Intrusion"]
TYPES = {
    "Vehicle + ANPR": ["Unauthorized vehicle", "Authorized vehicle"],
    "Helmet + Triple Riding": ["No helmet", "Triple riding"],
    "Threat Detection": ["Knife detected", "Gun detected", "Suspicious bag"],
    "Intrusion": ["Restricted zone entry", "Off-hours movement"],
}
SEVERITY = {"Unauthorized vehicle": "Medium", "Authorized vehicle": "Low", "No helmet": "Medium",
            "Triple riding": "Medium", "Knife detected": "High", "Gun detected": "High",
            "Suspicious bag": "High", "Restricted zone entry": "High", "Off-hours movement": "Medium"}
CAMERAS = ["Main Gate", "Parking A", "Library Road", "Hostel Gate", "Lab Block", "Back Gate"]


@st.cache_data
def demo_events(n=300):
    random.seed(7)
    rows = []
    for i in range(n):
        m = random.choice(MODULES)
        t = random.choice(TYPES[m])
        rows.append({
            "id": i + 1,
            "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 60 * 24 * 30)),
            "module": m, "camera": random.choice(CAMERAS), "event_type": t, "severity": SEVERITY[t],
            "plate": f"UK07{random.choice('ABCD')}{random.randint(1000, 9999)}" if m == "Vehicle + ANPR" else "",
            "evidence_url": "", "status": random.choice(["New", "Acknowledged", "Resolved"]),
        })
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)


def get_events():
    if BACKEND_URL:
        try:
            r = requests.get(f"{BACKEND_URL}/events", timeout=5)
            r.raise_for_status()
            d = pd.DataFrame(r.json())
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            return d
        except Exception as e:
            st.sidebar.error(f"Backend error: {e} (demo data)")
    return demo_events()


def get_cameras():
    if BACKEND_URL:
        try:
            return requests.get(f"{BACKEND_URL}/cameras", timeout=5).json()
        except Exception:
            pass
    random.seed(3)
    return [{"name": c, "status": "Offline" if c == "Back Gate" else "Online", "stream_url": ""} for c in CAMERAS]


def send_telegram(token, chat_id, text):
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": text}, timeout=10)
    return r.ok, r.text


df = get_events()
day_ago = datetime.now() - timedelta(hours=24)

# ---------------- Pages ----------------
@st.fragment(run_every=REFRESH)
def live_feeds():
    cams = get_cameras()
    recent = df[df["timestamp"] > day_ago]
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Cameras Online", f"{sum(c['status'] == 'Online' for c in cams)}/{len(cams)}", "#2E9E6B")
    with c2: kpi("Alerts (24h)", len(recent), BLUE)
    with c3: kpi("High Severity (24h)", int((recent['severity'] == 'High').sum()), "#D64545")
    with c4: kpi("Pending Action", int((df['status'] == 'New').sum()), "#F0A202")

    section("📹 Camera Feeds")
    cols = st.columns(3)
    for i, cam in enumerate(cams):
        on = cam["status"] == "Online"
        pill = f'<span class="pill" style="background:{"#2E9E6B" if on else "#D64545"}">{"● LIVE" if on else "● OFFLINE"}</span>'
        img = cam["stream_url"] or "https://placehold.co/640x360/0B1F3A/E8EEF7?text=" + cam["name"].replace(" ", "+")
        with cols[i % 3]:
            st.markdown(f'<div class="cam"><div class="name">{cam["name"]}{pill}</div>'
                        f'<img src="{img}" style="width:100%;border-radius:8px"></div>', unsafe_allow_html=True)

    section("🚨 Latest Alerts")
    for _, r in df.head(6).iterrows():
        c = SEV_COLOR[r["severity"]]
        st.markdown(
            f'<div class="alert" style="--c:{c}"><div><div class="t">{r["event_type"]}</div>'
            f'<div class="m">📍 {r["camera"]} &nbsp;•&nbsp; {r["module"]}</div></div>'
            f'<div class="r"><span class="pill" style="background:{c}">{r["severity"].upper()}</span>'
            f'<div class="m">{r["timestamp"]:%d %b, %H:%M}</div></div></div>', unsafe_allow_html=True)


banner(page.strip(), "Smart Campus Safety & Surveillance System — Control Room")

if page == PAGES[0]:
    live_feeds()

elif page == PAGES[1]:
    f1, f2, f3 = st.columns(3)
    mod = f1.multiselect("Module", MODULES, default=MODULES)
    sev = f2.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    stat = f3.multiselect("Status", ["New", "Acknowledged", "Resolved"], default=["New", "Acknowledged"])
    view = df[df["module"].isin(mod) & df["severity"].isin(sev) & df["status"].isin(stat)]
    section(f"Events ({len(view)})")
    st.dataframe(view.style.map(sev_style, subset=["severity"]), use_container_width=True,
                 hide_index=True, height=350)
    section("🔍 Check Evidence / Take Action")
    if len(view):
        eid = st.selectbox("Select Event ID", view["id"].head(50))
        row = df[df["id"] == eid].iloc[0]
        a, b = st.columns(2)
        with a:
            st.json({k: str(v) for k, v in row.to_dict().items()})
        with b:
            if row["evidence_url"]:
                st.image(row["evidence_url"])
            else:
                st.info("No evidence image yet (backend will provide evidence_url).")
            if st.button("✅ Acknowledge"):
                if BACKEND_URL:
                    requests.patch(f"{BACKEND_URL}/events/{eid}", json={"status": "Acknowledged"})
                st.success(f"Event {eid} acknowledged")

elif page == PAGES[2]:
    v = df[df["module"] == "Vehicle + ANPR"].copy()
    v["access"] = v["event_type"].map({"Authorized vehicle": "Authorized", "Unauthorized vehicle": "Unauthorized"})
    c1, c2, c3 = st.columns(3)
    with c1: kpi("Total Entries", len(v))
    with c2: kpi("Authorized", int((v["access"] == "Authorized").sum()), "#2E9E6B")
    with c3: kpi("Unauthorized", int((v["access"] == "Unauthorized").sum()), "#D64545")
    q = st.text_input("🔎 Search plate number")
    if q:
        v = v[v["plate"].str.contains(q.upper())]
    left, right = st.columns([3, 2])
    left.dataframe(v[["timestamp", "plate", "camera", "access"]], use_container_width=True, hide_index=True, height=400)
    right.plotly_chart(style_layout(px.pie(v, names="access", hole=.55, title="Access Split",
                       color="access", color_discrete_map={"Authorized": "#2E9E6B", "Unauthorized": "#D64545"})),
                       use_container_width=True)

elif page == PAGES[3]:
    period = st.radio("Period", ["Daily", "Monthly"], horizontal=True)
    d = df.copy()
    d["period"] = d["timestamp"].dt.strftime("%Y-%m-%d" if period == "Daily" else "%Y-%m")
    trend = d.groupby(["period", "module"]).size().reset_index(name="count")
    st.plotly_chart(style_layout(px.bar(trend, x="period", y="count", color="module",
                    title=f"{period} Events by Module", color_discrete_sequence=MOD_COLORS)),
                    use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(style_layout(px.pie(d, names="severity", hole=.55, title="Severity Split", color="severity",
                    color_discrete_map=SEV_COLOR)), use_container_width=True)
    d["hour"] = d["timestamp"].dt.hour
    c2.plotly_chart(style_layout(px.histogram(d, x="hour", nbins=24, title="Events by Hour of Day",
                    color_discrete_sequence=[BLUE])), use_container_width=True)
    st.download_button("⬇️ Download Report (CSV)", d.to_csv(index=False), "campus_report.csv")

elif page == PAGES[4]:
    token = st.text_input("Bot Token", type="password")
    chat_id = st.text_input("Chat ID")
    if st.button("📨 Send Test Message"):
        ok, resp = send_telegram(token, chat_id, "✅ Smart Campus Safety: test alert")
        st.success("Sent!") if ok else st.error(resp)
    section("High-Severity Alerts Pending")
    high = df[(df["severity"] == "High") & (df["status"] == "New")].head(5)
    st.dataframe(high[["timestamp", "camera", "event_type"]], hide_index=True, use_container_width=True)
    if st.button("🚨 Send These to Telegram"):
        for _, r in high.iterrows():
            send_telegram(token, chat_id,
                          f"🚨 {r['event_type']}\n📍 {r['camera']}\n🕒 {r['timestamp']:%d %b %H:%M}")
        st.success("Alerts sent")
