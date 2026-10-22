"""Shared presentation only; all user content uses native Streamlit elements."""

CSS = """
<style>
:root { color-scheme: dark; }
.stApp { background: #0b1220; }
.block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #101a2c; border-right: 1px solid #23314a; }
h1, h2, h3 { letter-spacing: -.025em; }
h1 { font-size: 2.2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.05rem !important; }
[data-testid="stMetric"] { background: #121d30; border: 1px solid #26364c;
    border-radius: 12px; padding: 16px 19px; min-height: 110px; }
[data-testid="stMetricLabel"] { color: #9cadc4; }
[data-testid="stMetricValue"] { font-size: 1.8rem; }
.eyebrow { color: #5dd6c0; font-size: .73rem; font-weight: 700;
    letter-spacing: .18em; margin-bottom: .65rem; }
.hero-copy { color: #aebbd0; max-width: 720px; font-size: 1.06rem; line-height: 1.7; }
.project-title { color: #e8eef8; max-width: 900px; font-size: 1.38rem;
    line-height: 1.35; font-weight: 650; margin: -.4rem 0 .7rem; }
.project-tagline { color: #5dd6c0; font-size: 1rem; font-weight: 600;
    margin-bottom: .8rem; }
.brand { font-size: 1.6rem; font-weight: 800; letter-spacing: .02em; color: #e8eef8; }
.brand-dot { color: #5dd6c0; }
.brand-subtitle { color: #95a6bf; font-size: .68rem; letter-spacing: .13em; margin-top: -.3rem; }
.step { background: #121d30; border: 1px solid #26364c; border-radius: 12px;
    padding: 22px; min-height: 162px; margin: .4rem 0 1rem; }
.step-num { color: #5dd6c0; font-size: .72rem; font-weight: 700; letter-spacing: .12em; }
.step-title { color: #e6edf7; font-weight: 650; margin: 8px 0; }
.step-copy { color: #a9b8cb; font-size: .9rem; line-height: 1.55; }
[data-testid="stDownloadButton"] button, [data-testid="stButton"] button { border-radius: 8px; }
[data-testid="stAlert"] { border-radius: 10px; }
footer { visibility: hidden; }
</style>
"""
