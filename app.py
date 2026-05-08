"""Matiks AI-first short-form content operations prototype.

Run with:
    streamlit run app.py

This app is intentionally lightweight: it demonstrates the operating system behind
10+ faceless/AI-UGC Instagram channels without requiring paid model API keys. The
same interfaces can be connected to OpenAI, Runway, HeyGen, ElevenLabs, CapCut,
Buffer/Metricool, and Meta Graph API in production.
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Matiks AI Content Engine",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container { padding-top: 1.6rem; max-width: 1280px; }
    .hero-card {
        padding: 1.35rem 1.55rem;
        border-radius: 1.2rem;
        background: radial-gradient(circle at top left, #22d3ee 0, transparent 28%),
                    linear-gradient(135deg, #0f172a 0%, #1e3a8a 52%, #7c2d12 100%);
        color: white;
        box-shadow: 0 18px 44px rgba(15, 23, 42, 0.22);
        margin-bottom: 1rem;
    }
    .hero-card h1 { margin-bottom: 0.25rem; }
    .hero-card p { opacity: 0.94; margin-bottom: 0; font-size: 1.04rem; }
    .system-card {
        padding: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.32);
        border-radius: 1rem;
        background: rgba(248, 250, 252, 0.78);
        min-height: 8.5rem;
    }
    .script-card {
        padding: 1rem;
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        border-radius: 0.8rem;
        margin-bottom: 0.85rem;
    }
    .danger { color: #b91c1c; font-weight: 700; }
    .good { color: #047857; font-weight: 700; }
    .pill {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        background: #dbeafe;
        color: #1e40af;
        font-size: 0.84rem;
        font-weight: 700;
        margin: 0.1rem 0.15rem 0.1rem 0;
    }
</style>
"""


@dataclass(frozen=True)
class Channel:
    """A channel configuration managed by one operator."""

    name: str
    niche: str
    format_type: str
    audience: str
    offer: str


CASE_STUDIES = {
    "Faceless AI content page": {
        "target": "An AI-history / dark-facts faceless Reels page using synthetic b-roll, captions, VO, and curiosity hooks.",
        "workflow": [
            "Mine Reddit, YouTube Shorts, Google Trends, TikTok Creative Center, and competitor reels for high-retention topics.",
            "Cluster topics into repeatable series: shocking origin, 3-part list, before/after, myth vs fact, and visual countdown.",
            "Generate 8-12 hook/script variants, score them for novelty, tension, visualizability, and compliance risk.",
            "Generate visuals with image/video models, pair with TTS, beat-cut scenes every 1.0-1.8 seconds, and burn subtitles.",
            "Schedule two posts/day, tag scripts by premise and hook, then recycle winners into adjacent niches.",
        ],
        "likely_stack": "ChatGPT/Claude for topic expansion, Perplexity/Serp for research, Midjourney/Flux + Runway/Pika/Kling for visuals, ElevenLabs for VO, CapCut/Descript/FFmpeg for assembly, Metricool/Buffer for scheduling, Sheets/Airtable for tracking.",
        "bottlenecks": "Original fact verification, consistent visual style, render queue time, avoiding repetitive hooks, and converting analytics into new scripts quickly.",
        "scale_play": "Separate creative strategy from production: one operator approves topic clusters and guardrails; agents generate briefs, scripts, prompts, captions, and edit decision lists for batch rendering.",
    },
    "AI UGC scaling brand": {
        "target": "A DTC consumable/beauty brand testing AI UGC-style Reels and paid social variants around testimonials, demos, and objections.",
        "workflow": [
            "Pull product reviews, Amazon/Shopify FAQs, creator comments, and competitor ads into an objection/benefit database.",
            "Generate UGC briefs by persona: skeptical buyer, busy parent, fitness beginner, office worker, creator comparison.",
            "Create 30-80 script variants per offer with different first lines, proof mechanisms, CTAs, and scene orders.",
            "Produce AI-avatar or human-edited UGC cuts; keep product claims compliant and avoid fake testimonial specifics.",
            "Launch as organic dark posts or ads, read 3-sec hold/CTR/CVR/comment sentiment, and mutate winners weekly.",
        ],
        "likely_stack": "Review scrapers, Airtable, ChatGPT/Claude, Arcads/Creatify/HeyGen/Akool for UGC avatars, CapCut templates, Motion/Foreplay-style ad libraries, Meta Ads + Northbeam/Triple Whale analytics.",
        "bottlenecks": "Trust erosion from uncanny avatars, claim review, product-shot consistency, asset rights, and too many variants without clean naming/analytics taxonomy.",
        "scale_play": "Treat AI UGC as a testing layer, not final brand truth: generate many low-cost angles, identify hooks that hold attention, then reshoot or polish the winners for higher-trust distribution.",
    },
}

DEFAULT_CHANNELS = [
    Channel("HistoryPulse", "history", "faceless facts", "curious 18-34", "follow for daily hidden history"),
    Channel("MoneyMinds", "personal finance", "avatar explainer", "new earners", "save this checklist"),
    Channel("FitMinute", "fitness", "routine demo", "busy beginners", "try the 7-day plan"),
    Channel("MindFuel", "self improvement", "text + b-roll", "ambitious students", "comment RESET"),
    Channel("PetOddities", "pets", "listicle", "animal lovers", "share with a pet friend"),
    Channel("FoodHacksAI", "food hacks", "hands demo", "home cooks", "save for dinner"),
    Channel("CareerLift", "career", "green-screen explainer", "job switchers", "download the template"),
    Channel("TravelTiny", "travel", "cinematic list", "budget travelers", "follow for micro-itineraries"),
    Channel("BeautyLab", "beauty", "AI UGC review", "skincare shoppers", "tap to compare routines"),
    Channel("TechSnap", "AI tools", "screen-record tutorial", "solo founders", "try the workflow"),
]

HOOKS = [
    "Nobody tells you this about {niche}.",
    "I tested the weirdest {niche} advice so you do not have to.",
    "This {niche} mistake costs people more than they think.",
    "Here is the 15-second version of {niche} that actually matters.",
    "If you only remember one {niche} rule, make it this.",
]

VISUAL_BEATS = [
    "0-2s: pattern interrupt close-up / bold text hook",
    "2-6s: proof visual, chart, or before-state",
    "6-12s: three rapid cuts explaining the mechanism",
    "12-18s: payoff, example, or mini transformation",
    "18-24s: CTA, save prompt, and next-video teaser",
]

CTA_PATTERNS = [
    "Save this before you scroll.",
    "Comment 'SYSTEM' and I will turn this into a checklist.",
    "Follow for the next part tomorrow.",
    "Send this to someone who needs the shortcut.",
]


def stable_random(*parts: str) -> random.Random:
    """Return a deterministic random generator so the demo is reproducible."""
    seed = int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16)
    return random.Random(seed)


def make_topic(channel: Channel, day_offset: int, reel_number: int) -> str:
    """Create a niche-specific topic without requiring a live LLM key."""
    rng = stable_random(channel.name, str(day_offset), str(reel_number))
    templates = [
        "the hidden pattern behind {niche} beginners miss",
        "3 counterintuitive {niche} lessons from high performers",
        "a simple {niche} system you can use in under 10 minutes",
        "the biggest {niche} myth that keeps spreading",
        "a before/after breakdown of a viral {niche} result",
    ]
    return rng.choice(templates).format(niche=channel.niche)


def score_idea(channel: Channel, topic: str) -> dict[str, int]:
    """Score an idea using operational heuristics that mimic an AI triage agent."""
    rng = stable_random(channel.name, topic)
    return {
        "hook_score": rng.randint(72, 98),
        "visual_score": rng.randint(66, 96),
        "production_effort": rng.randint(1, 5),
        "claim_risk": rng.randint(1, 5),
    }


def generate_script(channel: Channel, publish_day: date, reel_number: int) -> dict[str, object]:
    """Generate a full short-form content packet for one channel/reel."""
    topic = make_topic(channel, (publish_day - date.today()).days, reel_number)
    scores = score_idea(channel, topic)
    rng = stable_random(channel.name, topic, str(reel_number))
    hook = rng.choice(HOOKS).format(niche=channel.niche)
    cta = rng.choice(CTA_PATTERNS)
    proof = f"Use one concrete example from {channel.niche}; avoid unverifiable claims."
    script = (
        f"{hook}\n"
        f"Most people see {topic} as random, but there is a repeatable system. "
        f"Step one: isolate the trigger. Step two: show the contrast. Step three: give a tiny action the viewer can try today. "
        f"{proof} {cta}"
    )
    asset_prompt = (
        f"Create vertical 9:16 {channel.format_type} reel assets for @{channel.name}: "
        f"high-retention visuals about {topic}, fast cuts, large readable captions, consistent blue/orange brand accents."
    )
    return {
        "channel": channel.name,
        "niche": channel.niche,
        "format": channel.format_type,
        "audience": channel.audience,
        "publish_date": publish_day.isoformat(),
        "slot": "AM" if reel_number == 1 else "PM",
        "topic": topic,
        "hook": hook,
        "script": script,
        "visual_beats": " | ".join(VISUAL_BEATS),
        "asset_prompt": asset_prompt,
        "caption": f"{hook} {cta} #{channel.niche.replace(' ', '')} #reels #aitools",
        "cta": channel.offer,
        **scores,
        "priority": scores["hook_score"] + scores["visual_score"] - (scores["production_effort"] * 6) - (scores["claim_risk"] * 8),
        "status": "Needs human approval" if scores["claim_risk"] >= 4 else "Ready for generation",
    }


def build_content_calendar(channels: Iterable[Channel], days: int, reels_per_day: int) -> pd.DataFrame:
    """Build a cross-channel production calendar."""
    rows = []
    for day_offset in range(days):
        publish_day = date.today() + timedelta(days=day_offset)
        for channel in channels:
            for reel_number in range(1, reels_per_day + 1):
                rows.append(generate_script(channel, publish_day, reel_number))
    return pd.DataFrame(rows).sort_values(["publish_date", "channel", "slot"]).reset_index(drop=True)


def simulate_analytics(calendar: pd.DataFrame) -> pd.DataFrame:
    """Generate analytics and recommendations as a stand-in for Meta/Metricool ingestion."""
    rows = []
    for _, row in calendar.iterrows():
        rng = stable_random(row["channel"], row["topic"], row["slot"], "analytics")
        views = rng.randint(900, 125000)
        hold_rate = round(rng.uniform(0.24, 0.71), 3)
        saves = int(views * rng.uniform(0.004, 0.038))
        shares = int(views * rng.uniform(0.002, 0.026))
        comments = int(views * rng.uniform(0.001, 0.012))
        score = round((hold_rate * 100) + (saves / max(views, 1) * 700) + (shares / max(views, 1) * 900), 1)
        rows.append(
            {
                "channel": row["channel"],
                "niche": row["niche"],
                "topic": row["topic"],
                "hook": row["hook"],
                "views": views,
                "3s_hold_rate": hold_rate,
                "saves": saves,
                "shares": shares,
                "comments": comments,
                "winner_score": score,
                "recommendation": recommendation(score, hold_rate, row["hook"]),
            }
        )
    return pd.DataFrame(rows).sort_values("winner_score", ascending=False).reset_index(drop=True)


def recommendation(score: float, hold_rate: float, hook: str) -> str:
    """Turn metrics into next actions for the production agent."""
    if score >= 85:
        return f"Scale: create 5 variants of this hook family: '{hook[:54]}...'"
    if hold_rate < 0.34:
        return "Rewrite first 2 seconds; hook is not stopping the scroll."
    if score < 55:
        return "Kill angle; archive visuals and reuse only if niche demand changes."
    return "Iterate: keep topic, change payoff order and CTA."


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a dataframe to CSV bytes for Streamlit downloads."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL)
    return buffer.getvalue().encode("utf-8")


def render_case_studies() -> None:
    """Render operational reverse-engineering for the hiring assignment."""
    st.header("Part 1 — Reverse-engineered AI content machines")
    for title, details in CASE_STUDIES.items():
        with st.expander(title, expanded=True):
            st.markdown(f"**Target analyzed:** {details['target']}")
            st.markdown("**End-to-end workflow**")
            st.markdown("\n".join(f"{index}. {step}" for index, step in enumerate(details["workflow"], start=1)))
            st.markdown(f"**Likely tools/models/agents:** {details['likely_stack']}")
            st.markdown(f"**Operational bottlenecks:** {details['bottlenecks']}")
            st.markdown(f"**How Matiks recreates/scales it:** {details['scale_play']}")


def render_workflow() -> None:
    """Show the proposed production workflow."""
    st.header("Part 2 — Mini content engine workflow")
    steps = [
        ("Ideation", "Trend/competitor/review ingestion creates topic clusters per niche."),
        ("Research", "AI agent summarizes sources, extracts claims, and flags verification risk."),
        ("Scripting", "Hook generator creates variants; scorer ranks by novelty, visualizability, effort, and claim risk."),
        ("Generation", "Prompts are routed to avatar, image/video, VO, subtitle, and editing templates."),
        ("Editing", "EDL/shot list defines cuts, overlays, captions, music, and CTA for batch rendering."),
        ("Posting", "Calendar exports to scheduler/API with slot, channel, caption, hashtags, and asset names."),
        ("Tracking", "Metrics are normalized by channel/topic/hook family."),
        ("Feedback", "Winners produce variants; losers are killed or rewritten automatically."),
    ]
    columns = st.columns(4)
    for index, (name, detail) in enumerate(steps):
        with columns[index % 4]:
            st.markdown(f"<div class='system-card'><h4>{index + 1}. {name}</h4><p>{detail}</p></div>", unsafe_allow_html=True)


def render_scale_math(calendar: pd.DataFrame, channel_count: int, reels_per_day: int) -> None:
    """Display operator leverage metrics."""
    daily_reels = channel_count * reels_per_day
    weekly_reels = daily_reels * 7
    st.subheader("Operator leverage")
    cols = st.columns(4)
    cols[0].metric("Channels", channel_count)
    cols[1].metric("Reels/day", daily_reels)
    cols[2].metric("Reels/week", weekly_reels)
    cols[3].metric("Human review queue", int((calendar["status"] == "Needs human approval").sum()))
    st.caption(
        "The operator is not writing every reel. They approve guardrails, review risky claims, and inspect analytics-derived winner variants."
    )


def render_script_cards(calendar: pd.DataFrame) -> None:
    """Show the generated final outputs."""
    st.subheader("Final output generated by the system")
    for _, row in calendar.head(5).iterrows():
        status_class = "danger" if row["status"] == "Needs human approval" else "good"
        st.markdown(
            f"""
            <div class='script-card'>
                <span class='pill'>@{row['channel']}</span>
                <span class='pill'>{row['niche']}</span>
                <span class='pill'>{row['publish_date']} {row['slot']}</span>
                <h4>{row['topic']}</h4>
                <p><strong>Hook:</strong> {row['hook']}</p>
                <p><strong>Script:</strong> {row['script']}</p>
                <p><strong>Asset prompt:</strong> {row['asset_prompt']}</p>
                <p><strong>Caption:</strong> {row['caption']}</p>
                <p class='{status_class}'>{row['status']} · Priority {row['priority']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> tuple[int, int, int, list[Channel]]:
    """Collect operator controls."""
    with st.sidebar:
        st.header("⚙️ Engine controls")
        channel_count = st.slider("Channels to manage", 1, len(DEFAULT_CHANNELS), 10)
        reels_per_day = st.slider("Reels per channel/day", 1, 4, 2)
        days = st.slider("Calendar horizon", 1, 14, 3)
        st.divider()
        st.markdown("**Tool routing in production**")
        st.write("Research: Perplexity/Serp/API scraper")
        st.write("LLM: GPT/Claude prompt agents")
        st.write("Visuals: Flux/Midjourney + Runway/Kling/Pika")
        st.write("VO/UGC: ElevenLabs + HeyGen/Arcads/Creatify")
        st.write("Assembly: FFmpeg/CapCut templates")
        st.write("Distribution: Buffer/Metricool/Meta Graph API")
        st.write("Analytics: Meta Insights + warehouse")
    return channel_count, reels_per_day, days, DEFAULT_CHANNELS[:channel_count]


def main() -> None:
    """Run the Streamlit application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class='hero-card'>
            <h1>⚙️ Matiks AI Content Engine</h1>
            <p>A working prototype for operating 10+ Instagram channels at 2+ reels/day with AI-first research, scripting, generation briefs, scheduling, analytics, and feedback loops.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    channel_count, reels_per_day, days, channels = render_sidebar()
    calendar = build_content_calendar(channels, days, reels_per_day)
    analytics = simulate_analytics(calendar)

    render_case_studies()
    st.divider()
    render_workflow()
    st.divider()
    render_scale_math(calendar, channel_count, reels_per_day)

    tab_calendar, tab_scripts, tab_analytics, tab_exports = st.tabs(
        ["Production calendar", "Generated scripts", "Analytics loop", "Exports"]
    )

    with tab_calendar:
        st.subheader("Calendar, routing, and QA queue")
        st.dataframe(
            calendar[
                [
                    "publish_date",
                    "slot",
                    "channel",
                    "niche",
                    "format",
                    "topic",
                    "hook_score",
                    "visual_score",
                    "production_effort",
                    "claim_risk",
                    "priority",
                    "status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_scripts:
        render_script_cards(calendar.sort_values("priority", ascending=False))

    with tab_analytics:
        st.subheader("Simulated feedback loop")
        st.dataframe(analytics.head(20), use_container_width=True, hide_index=True)
        top = analytics.iloc[0]
        st.success(
            f"Next automation: clone the winning hook from @{top['channel']} into five variants, then route two to generation and three to script QA."
        )

    with tab_exports:
        st.subheader("Operator handoff files")
        st.download_button(
            "Download production calendar CSV",
            data=dataframe_to_csv_bytes(calendar),
            file_name=f"matiks_calendar_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download analytics feedback CSV",
            data=dataframe_to_csv_bytes(analytics),
            file_name=f"matiks_analytics_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(
            """
            **How this scales with minimal manual effort**
            - One operator manages guardrails, approval thresholds, and exception queues instead of writing every script.
            - Channel-specific templates keep niche voice, format, CTA, and visual style consistent.
            - CSV/API exports connect directly to render workers, schedulers, and analytics warehouses.
            - The feedback loop promotes hook families that win on retention/saves/shares and kills low-signal ideas.
            """
        )


if __name__ == "__main__":
    main()
