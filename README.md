# Matiks AI Content Engine

A Streamlit prototype for the Matiks hiring assignment: an AI-first operating system for one operator to manage 10+ Instagram channels, two or more Reels per channel per day, and multiple faceless/AI-UGC content formats across niches.

## What this prototype demonstrates

The app is not a generic creator dashboard. It models the scalable workflow behind a content machine:

1. Reverse-engineer a faceless AI short-form page and an AI-UGC scaling brand.
2. Generate channel-specific topic ideas, hooks, scripts, captions, asset prompts, and scheduling metadata.
3. Score each concept by hook strength, visualizability, production effort, and claim risk.
4. Create an operator QA queue so humans review only risky/high-leverage decisions.
5. Simulate analytics ingestion and convert metrics into next-step recommendations.
6. Export production calendar and analytics CSV files for schedulers, render workers, or warehouses.

## Workflow covered

```text
Ideation → research → scripting → generation → editing → posting → tracking → feedback loop
```

In production, each stage would route to specialized tools:

- **Research:** Perplexity, Serp API, competitor libraries, review/comment scrapers.
- **Scripting:** GPT/Claude prompt agents with channel memory and claim checks.
- **Image/video:** Flux, Midjourney, Runway, Pika, Kling, or similar generators.
- **UGC/avatar:** HeyGen, Arcads, Creatify, Akool, or human-edited creator variants.
- **Voice/editing:** ElevenLabs, CapCut templates, Descript, FFmpeg workers.
- **Posting:** Buffer, Metricool, Meta Graph API, or n8n/Zapier schedulers.
- **Analytics:** Meta Insights, ad dashboards, Google Sheets/Airtable, and a warehouse.

## How it scales

The prototype defaults to 10 channels and two Reels per channel per day. The operator is not manually creating 20 daily videos. Instead, they manage:

- channel templates and niche guardrails;
- generated briefs and scripts;
- claim-risk approvals;
- render/scheduling exports;
- analytics-driven winner cloning and loser pruning.

This gives one operator a queue-based workflow for running many channels simultaneously.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the local Streamlit URL, usually `http://localhost:8501`.

## Deliverable format

This repository can be submitted as the working prototype demo. A Loom walkthrough can show:

1. The reverse-engineered systems in Part 1.
2. The generated production calendar.
3. The script/asset-prompt outputs.
4. The analytics feedback loop.
5. The CSV exports that would feed rendering, posting, and reporting automations.
