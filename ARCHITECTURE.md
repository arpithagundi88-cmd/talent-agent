# Architecture & Scoring Logic — Talent Scout AI

## System Overview

Talent Scout AI is a single-file Streamlit application (`talent_scout.py`) that implements a 6-stage pipeline to automate recruiter outreach — from JD parsing to a ranked candidate shortlist.

---

## Pipeline Stages

### Stage 1 — JD Parser (1 LLM call)

**Input:** Raw job description text (any format)

**Process:** A structured prompt instructs the LLM to extract:
- `role` — job title
- `required_skills` — mandatory technical skills
- `preferred_skills` — optional/nice-to-have skills
- `location` — city/region
- `experience` — minimum years (integer)

**Output:** Python dict — used as the scoring reference for all candidates

**LLM:** `meta/llama-3.3-70b-instruct` via NVIDIA NIM API  
**Tokens:** ~400 max output  
**Fallback:** Returns empty defaults on parse error — app continues gracefully

---

### Stage 2 — Match Scorer (0 LLM calls)

**Input:** Parsed JD + entire candidate pool (SQLite)

**Process:** Pure Python rule-based scoring across 5 weighted factors:

```
Total Match Score = req_pts + pref_pts + exp_pts + role_pts + loc_pts
Maximum possible = 45 + 15 + 20 + 15 + 5 = 100
```

#### Factor 1: Required Skills (max 45 pts)
```python
ratio = matched_required / total_required
pts = (ratio ** 0.75) * 45
```
Uses a **power-curve** (`^0.75`) instead of linear scaling — this penalises partial matches more heavily. A candidate matching 2/4 skills gets `(0.5^0.75) * 45 = 26.9` rather than `0.5 * 45 = 22.5`.

#### Factor 2: Preferred Skills (max 15 pts)
```python
pts = (matched_preferred / total_preferred) * 15
```
Linear ratio — preferred skills are a bonus, not a requirement.

#### Factor 3: Experience (max 20 pts) — 3-tier
```python
if candidate_exp >= required_exp:          pts = 20  # "full"
elif candidate_exp >= required_exp - 1:    pts = 14  # "partial" (70%)
else:                                      pts = 0   # "miss"
```
The `exp_tier` label (`full` / `partial` / `miss`) is surfaced in the UI for explainability.

#### Factor 4: Role Alignment (max 15 pts)
```python
# Keyword overlap between JD role title and candidate's current role
# Stop words filtered: {and, or, of, the, a, an, in, at, for, to}
overlap = matching_keywords / total_jd_keywords
pts = overlap * 15
```

#### Factor 5: Location (max 5 pts)
```python
if no_location_specified:  pts = 3   # neutral
elif exact_match:          pts = 5   # full
else:                      pts = 2   # different city
```

**Output:** All candidates ranked by match score; top 3 proceed to Stage 3.

**Why rule-based?** Scoring the entire pool with LLM calls would be expensive and slow. Rule-based scoring is instant, deterministic, explainable, and free — the LLM is reserved for tasks that genuinely need reasoning (conversation, interest analysis).

---

### Stage 3 — Live Recruiter Chatbot (up to 7 LLM calls per candidate)

**Input:** Top 3 candidates + parsed JD + recruiter's typed messages

**Process (per turn, max 3 turns):**

1. **Suggestion Generation (1 LLM call):**
   The LLM generates 2 recruiter message options — one direct/confident, one warm/curious — tailored to the candidate's profile, the JD, and the conversation history so far. Suggestions are cached in session state so they don't regenerate on every Streamlit rerun.

2. **Recruiter Input:**
   The recruiter clicks a suggestion (auto-fills the text area) or types their own message. They can edit the suggestion freely before sending.

3. **Candidate Reply (1 LLM call):**
   The LLM roleplays as the candidate using their profile (skills, experience, location, current role). Temperature = 0.85 ensures varied, non-deterministic replies. The prompt instructs the model to be realistic — show genuine interest for good fits, polite hesitation for mismatches.

**LLM calls per candidate:** 3 suggestions + 3 candidate replies = **6 calls**  
**Total for 3 candidates:** 18 calls + 1 JD parse = **~19 LLM calls per run**

---

### Stage 4 — Interest Scorer (1 LLM call per candidate)

**Input:** Full 3-turn conversation per candidate

**Process:** The LLM analyses the conversation and scores 4 dimensions:

| Dimension | Max | What It Captures |
|---|---|---|
| Openness | 25 | Is the candidate actively looking? Passive vs active job seeker signals |
| Role Alignment | 25 | Did the role excite them? Enthusiasm, follow-up questions, specific skill mentions |
| Location Fit | 25 | Comfort with the job location — relocation willingness, remote preference |
| Availability | 25 | Notice period signals, urgency, how soon they could start |

**Output:** `interest_score` (0–100) + per-dimension breakdown + `key_quote` + summary

**Why LLM here?** Interest signals in natural language are nuanced — hesitation, enthusiasm, specific word choices, questions asked. Rule-based scoring would miss these signals entirely.

---

### Stage 5 — Combined Ranking & Export

```
Final Score = (Match Score × match_weight) + (Interest Score × interest_weight)
Default:      (Match Score × 0.6)          + (Interest Score × 0.4)
```

The weight ratio is a real-time slider in the sidebar — recruiters can adjust based on whether they care more about technical fit or candidate enthusiasm.

Output:
- Ranked shortlist table with progress bars
- Per-candidate expandable cards (match breakdown + interest breakdown + full chat transcript)
- CSV export

---

## Data Layer

**SQLite** (`talent.db`) — auto-created on first run, seeded with 7 candidate profiles if empty.

```sql
CREATE TABLE candidates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    skills       TEXT NOT NULL,      -- comma-separated
    experience   INTEGER NOT NULL,   -- years
    location     TEXT NOT NULL,
    current_role TEXT NOT NULL
)
```

CRUD operations: view pool, add candidate (sidebar form), delete candidate (per-row button).

To reset: `rm talent.db` — the app recreates and reseeds on next run.

---

## API Layer — NVIDIA NIM

**Model:** `meta/llama-3.3-70b-instruct`  
**Endpoint:** `https://integrate.api.nvidia.com/v1/chat/completions`

### Rate Limit Handling
```
Limit per key:   40 requests/minute
Min gap enforced: 2.0 seconds per key
Keys supported:  up to 3 (NVIDIA_API_KEY_1/2/3)
Effective limit: up to 120 rpm with 3 keys
```

### Error Handling
| Status | Action |
|---|---|
| 429 | Rotate to next key; honour Retry-After header |
| 401 | Rotate to next key (invalid/expired) |
| 5xx | Wait 5s, retry same key |
| Timeout | Rotate to next key |
| All keys exhausted | Exponential backoff (10s → 20s → 60s max), max 3 cycles |

---