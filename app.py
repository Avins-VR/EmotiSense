import streamlit as st
import cv2
from emotion_model import detect_emotions
from PIL import Image
import numpy as np
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense · AI Emotion Intelligence",
    page_icon="😉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Injected CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #050810 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
    overflow-x: hidden;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ── Animated background grid ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,102,241,.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,.06) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Radial glow orbs ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 50% at 10% 20%, rgba(99,102,241,.12) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 90% 80%, rgba(236,72,153,.10) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(16,185,129,.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Main block ── */
.block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px !important;
    position: relative;
    z-index: 1;
}

/* ── HEADER ── */
.header-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.6rem 2.4rem;
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 20px;
    backdrop-filter: blur(20px);
    margin-bottom: 2.4rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 8px 24px rgba(99,102,241,.35);
}

.brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #f9a8d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.3px;
}

.brand-tagline {
    font-size: 11px;
    color: #64748b;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 2px;
}

.status-badge {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 16px;
    background: rgba(16,185,129,.1);
    border: 1px solid rgba(16,185,129,.25);
    border-radius: 999px;
    font-size: 13px;
    color: #34d399;
    font-weight: 500;
}

.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    animation: pulse-dot 1.8s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .6; transform: scale(1.3); }
}

/* ── LAYOUT GRID ── */
.layout-grid {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 1.6rem;
    align-items: start;
}

/* ── VIDEO PANEL ── */
.video-panel {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 24px;
    overflow: hidden;
    backdrop-filter: blur(10px);
    position: relative;
}

.video-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.2rem 1.6rem;
    border-bottom: 1px solid rgba(255,255,255,.06);
}

.video-label {
    font-family: 'Syne', sans-serif;
    font-size: 14px; font-weight: 700;
    color: #94a3b8;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.rec-badge {
    display: flex; align-items: center; gap: 6px;
    padding: 4px 10px;
    background: rgba(239,68,68,.12);
    border: 1px solid rgba(239,68,68,.25);
    border-radius: 999px;
    font-size: 11px; font-weight: 600;
    color: #f87171;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.rec-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #ef4444;
    animation: pulse-dot 1.2s ease-in-out infinite;
}

.video-inner { padding: 1.2rem; }

.idle-screen {
    aspect-ratio: 4/3;
    background: rgba(0,0,0,.4);
    border-radius: 16px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 16px;
    border: 2px dashed rgba(255,255,255,.06);
}

.idle-icon { font-size: 52px; opacity: .25; }

.idle-text {
    font-size: 15px; color: #475569; font-weight: 400;
    letter-spacing: .3px;
}

/* ── SIDEBAR PANEL ── */
.side-panel {
    display: flex; flex-direction: column; gap: 1.2rem;
}

/* ── CARD ── */
.card {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(10px);
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px; font-weight: 700;
    color: #475569;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── EMOTION DISPLAY ── */
.emotion-main {
    text-align: center;
    padding: 1.6rem 0 1rem;
}

.emotion-emoji {
    font-size: 56px;
    display: block;
    margin-bottom: 10px;
    filter: drop-shadow(0 4px 24px rgba(99,102,241,.4));
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-6px); }
}

.emotion-name {
    font-family: 'Syne', sans-serif;
    font-size: 28px; font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #f9a8d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.emotion-sublabel {
    font-size: 12px; color: #475569;
    margin-top: 4px; font-weight: 300;
    letter-spacing: .8px;
}

/* ── CONFIDENCE RING ── */
.conf-wrap {
    display: flex; align-items: center; gap: 16px;
    margin-top: 1.2rem;
    padding: 12px 16px;
    background: rgba(0,0,0,.25);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,.05);
}

.conf-label { font-size: 12px; color: #64748b; }

.conf-value {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 700;
    color: #a5b4fc;
}

.conf-bar-track {
    flex: 1; height: 6px;
    background: rgba(255,255,255,.06);
    border-radius: 999px; overflow: hidden;
}

.conf-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    box-shadow: 0 0 10px rgba(99,102,241,.5);
    transition: width .4s ease;
}

/* ── EMOTION BARS ── */
.emo-bar-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 10px;
}

.emo-bar-name {
    width: 70px; font-size: 12px; color: #94a3b8; font-weight: 500;
    text-transform: capitalize;
}

.emo-bar-track {
    flex: 1; height: 5px;
    background: rgba(255,255,255,.05);
    border-radius: 999px; overflow: hidden;
}

.emo-bar-fill {
    height: 100%; border-radius: 999px;
    transition: width .5s ease;
}

.emo-bar-pct {
    width: 38px; text-align: right;
    font-size: 11px; color: #475569;
    font-variant-numeric: tabular-nums;
}

/* ── STAT CARDS ── */
.stats-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .8rem;
    margin-top: .6rem;
}

.stat-box {
    background: rgba(0,0,0,.25);
    border: 1px solid rgba(255,255,255,.05);
    border-radius: 14px;
    padding: 14px;
    text-align: center;
}

.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 22px; font-weight: 700;
    color: #e2e8f0;
}

.stat-lbl {
    font-size: 11px; color: #475569;
    margin-top: 3px;
    letter-spacing: .6px;
    text-transform: uppercase;
}

/* ── TOGGLE CONTROL ── */
.stCheckbox { margin-bottom: 0 !important; }

.stCheckbox label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
    padding: 14px 28px !important;
    background: linear-gradient(135deg, rgba(99,102,241,.2), rgba(236,72,153,.15)) !important;
    border: 1px solid rgba(99,102,241,.35) !important;
    border-radius: 16px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #a5b4fc !important;
    cursor: pointer !important;
    transition: all .25s ease !important;
    letter-spacing: .4px !important;
    width: 100% !important;
}

.stCheckbox label:hover {
    background: linear-gradient(135deg, rgba(99,102,241,.3), rgba(236,72,153,.25)) !important;
    border-color: rgba(99,102,241,.6) !important;
    box-shadow: 0 0 28px rgba(99,102,241,.2) !important;
    transform: translateY(-1px) !important;
}

.stCheckbox > div { justify-content: center !important; }

/* ── IMAGE DISPLAY ── */
.stImage > img {
    border-radius: 16px !important;
    width: 100% !important;
}

/* ── STREAMLIT OVERRIDES ── */
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.element-container { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Emotion Meta (emoji + gradient color) ──────────────────────────────────
EMOTION_META = {
    "happy":     ("😊", "#facc15", "#fb923c"),
    "sad":       ("😢", "#60a5fa", "#818cf8"),
    "angry":     ("😠", "#f87171", "#fb923c"),
    "fear":      ("😨", "#c084fc", "#818cf8"),
    "surprise":  ("😮", "#34d399", "#60a5fa"),
    "disgust":   ("🤢", "#86efac", "#4ade80"),
    "neutral":   ("😐", "#94a3b8", "#cbd5e1"),
}

def get_meta(emotion: str):
    key = emotion.lower() if emotion else "neutral"
    return EMOTION_META.get(key, ("🤖", "#a5b4fc", "#f9a8d4"))

# ─── Session State ───────────────────────────────────────────────────────────
if "frame_count"    not in st.session_state: st.session_state.frame_count    = 0
if "face_count"     not in st.session_state: st.session_state.face_count     = 0
if "dominant_hist"  not in st.session_state: st.session_state.dominant_hist  = "—"

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
  <div class="brand">
    <div class="brand-icon">😉</div>
    <div>
      <div class="brand-name">EmotiSense</div>
      <div class="brand-tagline">AI Emotion Intelligence</div>
    </div>
  </div>
  <div class="status-badge">
    <div class="status-dot"></div>
    Model Ready · FER + MTCNN
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout: two columns ─────────────────────────────────────────────────────
left, right = st.columns([2.2, 1], gap="medium")

# ════════════ LEFT: VIDEO ════════════
with left:
    st.markdown("""
    <div class="video-panel">
      <div class="video-header">
        <span class="video-label">Live Feed</span>
        <span class="rec-badge"><span class="rec-dot"></span>REC</span>
      </div>
      <div class="video-inner">
    """, unsafe_allow_html=True)

    # Toggle
    run = st.checkbox("⏻  Activate Camera", key="cam_toggle")

    # Frame placeholder
    frame_slot = st.empty()

    if not run:
        frame_slot.markdown("""
        <div class="idle-screen">
          <span class="idle-icon">📷</span>
          <span class="idle-text">Activate the camera to begin analysis</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

# ════════════ RIGHT: SIDEBAR ════════════
with right:
    # Primary emotion card
    emotion_card   = st.empty()
    all_bars_card  = st.empty()
    stats_card     = st.empty()

    def render_idle_cards():
        emotion_card.markdown("""
        <div class="card">
          <div class="card-title">Detected Emotion</div>
          <div class="emotion-main">
            <span class="emotion-emoji">—</span>
            <div class="emotion-name" style="font-size:20px;color:#334155;">Awaiting Input</div>
            <div class="emotion-sublabel">No face in frame</div>
          </div>
          <div class="conf-wrap">
            <div><div class="conf-label">Confidence</div>
                 <div class="conf-value">0%</div></div>
            <div class="conf-bar-track">
              <div class="conf-bar-fill" style="width:0%"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        all_bars_card.markdown("""
        <div class="card">
          <div class="card-title">Emotion Breakdown</div>
          <div style="color:#334155;font-size:13px;text-align:center;padding:1rem 0;">
            No data yet
          </div>
        </div>
        """, unsafe_allow_html=True)

        stats_card.markdown("""
        <div class="card">
          <div class="card-title">Session Stats</div>
          <div class="stats-row">
            <div class="stat-box"><div class="stat-num">0</div>
              <div class="stat-lbl">Frames</div></div>
            <div class="stat-box"><div class="stat-num">0</div>
              <div class="stat-lbl">Faces</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    render_idle_cards()

# ─── Camera Loop ─────────────────────────────────────────────────────────────
if run:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while st.session_state.get("cam_toggle", False):
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️  Camera unavailable — check permissions.")
            break

        st.session_state.frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detect_emotions(rgb)

        detected_emotion = "neutral"
        confidence       = 0.0
        all_emotions     = {}

        if results:
            st.session_state.face_count += len(results)
            for res in results:
                x, y, w, h = res["box"]
                emo         = res["emotions"]
                dom         = max(emo, key=emo.get)
                conf        = emo[dom]

                if conf > confidence:
                    confidence       = conf
                    detected_emotion = dom
                    all_emotions     = emo
                    st.session_state.dominant_hist = dom.capitalize()

                # Draw bounding box + label
                emoji, c1, c2 = get_meta(dom)
                cv2.rectangle(rgb, (x, y), (x+w, y+h), (99, 102, 241), 2)
                lbl = f"{emoji} {dom.capitalize()}  {conf*100:.0f}%"
                (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
                cv2.rectangle(rgb, (x, y-lh-14), (x+lw+10, y), (20, 20, 40), -1)
                cv2.putText(rgb, lbl, (x+5, y-6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (165, 180, 252), 1, cv2.LINE_AA)

        # ── show frame ──
        frame_slot.image(rgb, channels="RGB", width="stretch")

        # ── emotion card ──
        emoji, c1, c2 = get_meta(detected_emotion)
        pct = confidence * 100
        emotion_card.markdown(f"""
        <div class="card">
          <div class="card-title">Detected Emotion</div>
          <div class="emotion-main">
            <span class="emotion-emoji">{emoji}</span>
            <div class="emotion-name">{detected_emotion.capitalize()}</div>
            <div class="emotion-sublabel">Primary expression</div>
          </div>
          <div class="conf-wrap">
            <div>
              <div class="conf-label">Confidence</div>
              <div class="conf-value">{pct:.0f}%</div>
            </div>
            <div class="conf-bar-track">
              <div class="conf-bar-fill" style="width:{pct:.1f}%"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── all-emotions bars ──
        if all_emotions:

            bar_colors = {
                "happy": "linear-gradient(90deg,#facc15,#fb923c)",
                "sad": "linear-gradient(90deg,#60a5fa,#818cf8)",
                "angry": "linear-gradient(90deg,#f87171,#fb923c)",
                "fear": "linear-gradient(90deg,#c084fc,#818cf8)",
                "surprise": "linear-gradient(90deg,#34d399,#60a5fa)",
                "disgust": "linear-gradient(90deg,#86efac,#4ade80)",
                "neutral": "linear-gradient(90deg,#94a3b8,#cbd5e1)"
            }

            bars_html = ""

            for emo, val in sorted(all_emotions.items(), key=lambda x: -x[1]):

                grad = bar_colors.get(
                    emo,
                    "linear-gradient(90deg,#a5b4fc,#f9a8d4)"
                )

                bars_html += (
                    '<div class="emo-bar-row">'
                    f'<span class="emo-bar-name">{emo}</span>'
                    '<div class="emo-bar-track">'
                    f'<div class="emo-bar-fill" '
                    f'style="width:{val*100:.1f}%;background:{grad};"></div>'
                    '</div>'
                    f'<span class="emo-bar-pct">{val*100:.0f}%</span>'
                    '</div>'
                )

            html_code = f"""
            <div class="card">
                <div class="card-title">Emotion Breakdown</div>
                {bars_html}
            </div>
            """

            all_bars_card.markdown(html_code, unsafe_allow_html=True)

        # ── session stats ──
        stats_card.markdown(f"""
        <div class="card">
          <div class="card-title">Session Stats</div>
          <div class="stats-row">
            <div class="stat-box">
              <div class="stat-num">{st.session_state.frame_count}</div>
              <div class="stat-lbl">Frames</div>
            </div>
            <div class="stat-box">
              <div class="stat-num">{st.session_state.face_count}</div>
              <div class="stat-lbl">Faces</div>
            </div>
          </div>
          <div style="margin-top:.8rem;padding:10px 14px;background:rgba(0,0,0,.25);
                      border-radius:12px;border:1px solid rgba(255,255,255,.05);">
            <div style="font-size:11px;color:#475569;text-transform:uppercase;
                        letter-spacing:1px;margin-bottom:4px;">Last Dominant</div>
            <div style="font-family:'Syne',sans-serif;font-size:16px;font-weight:700;
                        color:#a5b4fc;">{st.session_state.dominant_hist}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    cap.release()