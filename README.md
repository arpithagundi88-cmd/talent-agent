# 🎯 Talent Scout AI
### AI-Powered Talent Scouting & Engagement Agent
> Built for **Catalyst Hackathon** by Deccan AI · Submission by **Arpitha GS**

---

## 🚀 Live Demo
> _Add your deployed URL here after hosting (e.g. Streamlit Cloud)_

---

## 📌 What It Does

Talent Scout AI automates the full recruiter workflow — from parsing a job description to producing a ranked, explainable shortlist — in one seamless pipeline:

1. **JD Parsing** — paste any job description, the LLM extracts role, required skills, preferred skills, location, and minimum experience
2. **Match Scoring** — every candidate in the pool is scored across 5 weighted factors with zero LLM calls (fast, cost-efficient)
3. **Top-3 Selection** — only the best-matched candidates proceed to the engagement phase
4. **Live Recruiter Chatbot** — the recruiter types real messages (up to 3 turns); the LLM replies as each candidate based on their actual profile
5. **AI Suggestions** — 2 context-aware message suggestions per turn (one direct, one warm); click to auto-fill or type your own
6. **Interest Scoring** — after 3 turns, the LLM analyses the full conversation across 4 dimensions and produces an interest score (0–100)
7. **Combined Ranking** — final score = 60% Match + 40% Interest, giving recruiters an immediately actionable shortlist with full explainability

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TALENT SCOUT AI                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  Job         │    │         LLM (NVIDIA NIM)             │  │
│  │  Description │───▶│  meta/llama-3.3-70b-instruct         │  │
│  │  (Text Input)│    │                                      │  │
│  └──────────────┘    │  ┌─────────────────────────────────┐ │  │
│                      │  │  Step 1: JD Parser              │ │  │
│                      │  │  → role, skills, exp, location  │ │  │
│                      │  └──────────────┬──────────────────┘ │  │
│                      └─────────────────│────────────────────┘  │
│                                        │                        │
│  ┌─────────────────────────────────────▼──────────────────────┐ │
│  │           Step 2: Rule-Based Match Scorer (0 LLM calls)    │ │
│  │                                                            │ │
│  │  Required Skills  ████████████████████  45 pts (power-curve│ │
│  │  Preferred Skills ██████               15 pts (linear)    │ │
│  │  Experience       ████████             20 pts (3-tier)    │ │
│  │  Role Alignment   ██████               15 pts (keyword)   │ │
│  │  Location         ██                    5 pts (exact/diff) │ │
│  │                                                            │ │
│  │  → Scores entire candidate pool → Top 3 selected          │ │
│  └─────────────────────────────────────┬──────────────────────┘ │
│                                        │                        │
│  ┌─────────────────────────────────────▼──────────────────────┐ │
│  │        Step 3: Live Recruiter Chatbot (Top 3 only)         │ │
│  │                                                            │ │
│  │  Turn 1 ──▶ [AI Suggestions A/B] ──▶ Recruiter types/picks│ │
│  │  Turn 2 ──▶ [AI Suggestions A/B] ──▶ Recruiter types/picks│ │
│  │  Turn 3 ──▶ [AI Suggestions A/B] ──▶ Recruiter types/picks│ │
│  │                │                                           │ │
│  │                ▼                                           │ │
│  │         LLM replies as candidate                          │ │
│  │         (profile-aware, randomised, realistic)            │ │
│  └─────────────────────────────────────┬──────────────────────┘ │
│                                        │                        │
│  ┌─────────────────────────────────────▼──────────────────────┐ │
│  │        Step 4: Interest Scorer (1 LLM call / candidate)    │ │
│  │                                                            │ │
│  │  Openness        0–25  (actively looking?)                │ │
│  │  Role Alignment  0–25  (excited about this role?)         │ │
│  │  Location Fit    0–25  (comfortable with location?)       │ │
│  │  Availability    0–25  (can join soon?)                   │ │
│  └─────────────────────────────────────┬──────────────────────┘ │
│                                        │                        │
│  ┌─────────────────────────────────────▼──────────────────────┐ │
│  │        Step 5: Combined Ranking & Export                   │ │
│  │                                                            │ │
│  │  Final Score = Match × 0.6  +  Interest × 0.4             │ │
│  │  → Ranked shortlist table with progress bars              │ │
│  │  → Per-candidate breakdown (match + interest + chat)      │ │
│  │  → CSV export                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐  │
│  │  SQLite DB           │    │  NVIDIA NIM API              │  │
│  │  talent.db           │    │  Up to 3 API keys            │  │
│  │  Candidate pool CRUD │    │  Key rotation on 429         │  │
│  │  Seeded with 7 devs  │    │  2s rate-limit enforcement   │  │
│  └──────────────────────┘    └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Scoring Logic

### Match Score (0–100) — Rule-Based, Zero LLM Calls

| Factor | Weight | Method |
|---|---|---|
| Required Skills | 45 pts | Power-curve ratio `(matched/total)^0.75` — penalises partial matches |
| Preferred Skills | 15 pts | Linear ratio |
| Experience | 20 pts | 3-tier: Full (≥ req) = 20, Partial (req−1) = 14, Miss = 0 |
| Role Alignment | 15 pts | Keyword overlap between JD role title and candidate's current role |
| Location | 5 pts | Exact match = 5, different = 2, no preference = 3 |

### Interest Score (0–100) — LLM-Analysed from Conversation

| Factor | Weight | What It Measures |
|---|---|---|
| Openness | 25 pts | How actively is the candidate looking? |
| Role Alignment | 25 pts | Does this specific role excite them? |
| Location Fit | 25 pts | Are they comfortable with the location? |
| Availability | 25 pts | How soon could they realistically join? |

### Final Combined Score
```
Final = (Match Score × 0.6) + (Interest Score × 0.4)
```
Weight ratio is adjustable via the sidebar slider (0.4–0.8 match weight).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (Python) |
| LLM | NVIDIA NIM — `meta/llama-3.3-70b-instruct` |
| Database | SQLite (local, via `sqlite3`) |
| HTTP Client | `requests` |
| Data | `pandas` |
| Config | `python-dotenv` |

---

## 📦 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/talent-scout-ai.git
cd talent-scout-ai
```

### 2. Install dependencies
```bash
pip install streamlit requests python-dotenv pandas
```

### 3. Set up your NVIDIA API key(s)
Create a `.env` file in the project root:
```env
NVIDIA_API_KEY_1=nvapi-xxxxxxxxxxxxxxxxxxxx
NVIDIA_API_KEY_2=nvapi-xxxxxxxxxxxxxxxxxxxx   # optional
NVIDIA_API_KEY_3=nvapi-xxxxxxxxxxxxxxxxxxxx   # optional
```
Get a free key at: https://build.nvidia.com

> **Tip:** Adding 2–3 keys increases your effective rate limit from 40 rpm to 80–120 rpm combined.

### 4. Run the app
```bash
streamlit run talent_scout.py
```

The app opens at `http://localhost:8501`

### 5. Fresh start (reset candidate DB)
```bash
rm talent.db
streamlit run talent_scout.py
```

---

## 📂 Project Structure

```
talent-scout-ai/
├── talent_scout.py      # Main app — entire pipeline in one file
├── .env                 # Your API keys (not committed)
├── .env.example         # Template for .env
├── .gitignore
├── requirements.txt
├── README.md
├── ARCHITECTURE.md      # Detailed architecture + scoring docs
└── sample_output/
    ├── sample_jd.txt    # Example job description
    └── sample_output.csv  # Example ranked shortlist export
```

---

## 🧪 Sample Input

**Job Description:**
```
We are hiring a Senior Python Backend Developer based in Bangalore.
The ideal candidate has 4+ years of experience with FastAPI and AWS.
Preferred: Docker, PostgreSQL. Must be comfortable owning backend
services end-to-end in a fast-paced startup environment.
```

**Expected top candidates:** Rahul (FastAPI + AWS + Bangalore), Meera (FastAPI + PostgreSQL + Bangalore)

---

## 💡 Key Design Decisions

- **Zero LLM calls for match scoring** — the entire candidate pool is scored with pure Python logic, making it fast and cost-efficient regardless of pool size
- **Top-3 filter** — LLM engagement only happens for the top 3 candidates, keeping API usage within free-tier limits
- **Live recruiter chatbot** — unlike static simulation, the recruiter types real messages, making the interest assessment more authentic
- **AI suggestions with auto-fill** — reduces recruiter effort without removing control; suggestions are context-aware and turn-specific
- **Key rotation** — up to 3 NVIDIA keys rotate automatically on 429 errors with Retry-After header support
- **Adjustable weights** — match/interest weight ratio is a runtime slider, not hardcoded

---

## 🔑 API Rate Limit Handling

```
40 rpm per key × up to 3 keys = 120 rpm effective capacity
2.0s minimum gap enforced per key
Auto-rotation on 429 / 401 errors
Exponential backoff after all keys exhausted
Retry-After header respected
```

---

## 📊 Sample Output

| Rank | Name | Match /100 | Interest /100 | Final /100 | Interest Level |
|---|---|---|---|---|---|
| 1 | Rahul | 80.0 | 85 | 82.0 | Highly Interested |
| 2 | Meera | 71.3 | 72 | 71.6 | Interested |
| 3 | Arjun | 53.8 | 48 | 51.5 | Lukewarm |

---

## 👩‍💻 Author

**Arpitha GS** — Catalyst Hackathon 2025, Deccan AI
