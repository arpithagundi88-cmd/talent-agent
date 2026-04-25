"""
Talent Scout AI — AI-Powered Talent Scouting & Engagement Agent
================================================================
Run with:
    pip install streamlit requests python-dotenv pandas
    streamlit run talent_scout.py

Requires NVIDIA_API_KEY_1/2/3 in our .env file (supports up to 3 keys).

Pipeline:
    1. Parse JD              → structured fields via LLM
    2. Match scoring         → weighted 5-factor score (0–100) — I have used ZERO LLM calls here.
    3. Top-3 filter          → only top 3 by match score proceed to LLM
    4. LIVE chatbot          → recruiter sends real messages, LLM replies as candidate (max 3 turns)
    5. Interest scoring      → 4-factor score from full conversation (0–100)
    6. Combined ranking      → 60% match + 40% interest, sorted shortlist
"""

import json
import os
import sqlite3
import time
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Talent Scout AI", layout="wide", page_icon="🎯")

# ─────────────────────────────────────────────────────────────────────────────
# CSS — chat bubbles + live chat UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Chat bubbles ─────────────────────────────────────────────────────────── */
.chat-wrap { display:flex; flex-direction:column; gap:10px; padding:12px 0; }
.bubble-row-recruiter { display:flex; justify-content:flex-start; align-items:flex-end; gap:8px; }
.bubble-row-candidate { display:flex; justify-content:flex-end; align-items:flex-end; gap:8px; }
.avatar {
    width:32px; height:32px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:14px; flex-shrink:0; background:#e0e0e0;
}
.bubble { max-width:68%; padding:10px 14px; border-radius:18px; font-size:14px; line-height:1.5; }
.bubble-recruiter { background:#1D9E75; color:#fff; border-bottom-left-radius:4px; }
.bubble-candidate { background:#f0f0f0; color:#111; border-bottom-right-radius:4px; }
.bubble-label { font-size:11px; color:#888; margin-bottom:2px; }
.bubble-label-right { text-align:right; }

/* ── Live chat panel ──────────────────────────────────────────────────────── */
.live-chat-box {
    border: 2px solid #1D9E75;
    border-radius: 16px;
    padding: 16px;
    background: #f9fffe;
    margin-bottom: 12px;
}
.live-chat-header {
    font-weight: 700;
    font-size: 15px;
    color: #1D9E75;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.turn-counter {
    background: #1D9E75;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
}
.turn-counter-warn {
    background: #e67e22;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
}
.turn-counter-done {
    background: #7f8c8d;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
}
.candidate-typing {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: #f0f0f0;
    border-radius: 18px;
    font-size: 13px;
    color: #555;
    width: fit-content;
    margin-left: auto;
}
.hint-text {
    font-size: 12px;
    color: #888;
    font-style: italic;
    margin-top: 4px;
}
.suggestion-label {
    font-size: 12px;
    font-weight: 600;
    color: #555;
    margin-bottom: 6px;
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NVIDIA API — Key pool + rotation + retry
# I have used 3 API keys for safety purpose , since only 40rpm is supported by single NVIDIA key.
# ─────────────────────────────────────────────────────────────────────────────
_ALL_KEYS = [
    k for k in [
        os.getenv("NVIDIA_API_KEY_1"),
        os.getenv("NVIDIA_API_KEY_2"),
        os.getenv("NVIDIA_API_KEY_3"),
    ]
    if k and k.strip()
]

_legacy = os.getenv("NVIDIA_API_KEY_1")
if _legacy and _legacy.strip() and _legacy not in _ALL_KEYS:
    _ALL_KEYS.append(_legacy)

_key_state = {"index": 0}
_last_call_time: dict[int, float] = {}
MIN_CALL_INTERVAL = 2.0
MAX_RETRIES = 3


def _current_key() -> str:
    return _ALL_KEYS[_key_state["index"]]


def _rotate_key(reason: str) -> bool:
    next_idx = _key_state["index"] + 1
    if next_idx < len(_ALL_KEYS):
        _key_state["index"] = next_idx
        st.toast(f"🔑 Rotating to key {next_idx + 1} ({reason})")
        return True
    else:
        _key_state["index"] = 0
        st.toast(f"⚠️ All {len(_ALL_KEYS)} keys tried ({reason})")
        return False


def _wait_for_rate_limit():
    idx = _key_state["index"]
    last = _last_call_time.get(idx, 0)
    elapsed = time.time() - last
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)
    _last_call_time[idx] = time.time()


def call_nvidia(prompt: str, max_tokens: int = 500) -> str:
    if not _ALL_KEYS:
        raise RuntimeError(
            "No NVIDIA API keys found.\n"
            "Add to your .env file:\n"
            "  NVIDIA_API_KEY_1=nvapi-...\n"
            "  NVIDIA_API_KEY_2=nvapi-...\n"
            "  NVIDIA_API_KEY_3=nvapi-..."
        )

    body = {
        "model": "meta/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.85,  # slightly higher for more varied candidate responses
    }

    backoff = 10

    for attempt in range(MAX_RETRIES):
        for _ in range(len(_ALL_KEYS)):
            _wait_for_rate_limit()
            headers = {
                "Authorization": f"Bearer {_current_key()}",
                "Content-Type": "application/json",
            }
            key_num = _key_state["index"] + 1

            try:
                r = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=45,
                )

                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip()

                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", 0))
                    st.toast(f"⏳ Key {key_num} hit 40rpm limit — rotating...")
                    rotated = _rotate_key("429")
                    if not rotated:
                        wait_time = max(retry_after, backoff)
                        st.toast(f"😴 All keys rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        backoff = min(backoff * 2, 60)
                        break
                    elif retry_after:
                        time.sleep(retry_after)
                    continue

                if r.status_code == 401:
                    st.toast(f"🔑 Key {key_num} auth failed — rotating...")
                    _rotate_key("401 invalid key")
                    continue

                if r.status_code >= 500:
                    st.toast(f"🔧 Server error {r.status_code} — retrying in 5s...")
                    time.sleep(5)
                    continue

                r.raise_for_status()

            except requests.exceptions.Timeout:
                st.toast(f"⏰ Key {key_num} timed out — rotating...")
                _rotate_key("timeout")
                continue

            except requests.exceptions.ConnectionError:
                st.toast("🌐 Connection error — retrying in 5s...")
                time.sleep(5)
                continue

        if attempt < MAX_RETRIES - 1:
            st.toast(f"♻️ Retry cycle {attempt + 2}/{MAX_RETRIES} in {backoff}s...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    raise RuntimeError(
        f"NVIDIA API unavailable after {MAX_RETRIES} retry cycles.\n"
        "Your 40 rpm quota across all keys is exhausted.\n"
        "Wait 1 minute then try again, or add more keys to .env."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Database- 8 seed candidate samples are added to the system for demo purpose.
# multiple other people can be added using add candidate facility.
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect("talent.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            skills TEXT NOT NULL,
            experience INTEGER NOT NULL,
            location TEXT NOT NULL,
            current_role TEXT NOT NULL
        )
    """)
    c.execute("SELECT COUNT(*) FROM candidates")
    if c.fetchone()[0] == 0:
        seed = [
            ("Rahul",  "Python, SQL, FastAPI, AWS",            4, "Bangalore", "Backend Developer"),
            ("Priya",  "Python, Django, SQL, Azure",           5, "Hyderabad", "Software Engineer"),
            ("Amit",   "Java, Spring, MySQL",                  6, "Pune",      "Senior Developer"),
            ("Sneha",  "Python, FastAPI, AWS, Docker",         3, "Chennai",   "Python Developer"),
            ("Kiran",  "React, Node.js, MongoDB",              4, "Bangalore", "Full Stack Developer"),
            ("Neha",   "Python, SQL, Machine Learning",        2, "Delhi",     "Data Analyst"),
            ("Arjun",  "Python, AWS, Docker, Kubernetes",      5, "Mumbai",    "DevOps Engineer"),
            ("Meera",  "Python, FastAPI, PostgreSQL",          4, "Bangalore", "Backend Engineer"),
        ]
        c.executemany(
            "INSERT INTO candidates (name, skills, experience, location, current_role) VALUES (?,?,?,?,?)",
            seed
        )
    conn.commit()
    conn.close()


def get_candidates() -> pd.DataFrame:
    conn = sqlite3.connect("talent.db")
    df = pd.read_sql_query("SELECT * FROM candidates ORDER BY id", conn)
    conn.close()
    return df


def add_candidate(name, skills, experience, location, current_role):
    conn = sqlite3.connect("talent.db")
    conn.execute(
        "INSERT INTO candidates (name, skills, experience, location, current_role) VALUES (?,?,?,?,?)",
        (name, skills, experience, location, current_role)
    )
    conn.commit()
    conn.close()


def delete_candidate(candidate_id: int):
    conn = sqlite3.connect("talent.db")
    conn.execute("DELETE FROM candidates WHERE id=?", (candidate_id,))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — JD Parser
# ─────────────────────────────────────────────────────────────────────────────
def parse_jd(jd: str) -> dict:
    prompt = f"""
You are an expert recruiter assistant.
Extract structured hiring information from the job description.
Return ONLY valid JSON — no markdown, no explanation:

{{
  "role": "",
  "required_skills": [],
  "preferred_skills": [],
  "location": "",
  "experience": 0
}}

Rules:
- required_skills = mandatory technical skills only
- preferred_skills = optional / nice-to-have skills only
- experience = minimum years as integer
- If missing: use empty string / [] / 0

Job Description:
{jd}
"""
    try:
        text = call_nvidia(prompt, max_tokens=400)
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return {
            "role":             data.get("role", ""),
            "required_skills":  data.get("required_skills", []),
            "preferred_skills": data.get("preferred_skills", []),
            "location":         data.get("location", ""),
            "experience":       int(data.get("experience", 0))
        }
    except Exception as e:
        st.error(f"JD parsing error: {e}")
        return {"role": "", "required_skills": [], "preferred_skills": [], "location": "", "experience": 0}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Match Score Engine (zero LLM calls)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_match_score(parsed_jd: dict, row: pd.Series) -> dict:
    weights = {"required_skills": 45, "preferred_skills": 15,
               "experience": 20, "role": 15, "location": 5}
    score = 0
    candidate_skills = row["skills"].lower()

    my_required_skills = parsed_jd["required_skills"]
    matched_req, req_pts = [], 0
    if my_required_skills:
        matched_req = [s for s in my_required_skills if s.lower() in candidate_skills]
        ratio = len(matched_req) / len(my_required_skills)
        req_pts = round((ratio ** 0.75) * weights["required_skills"], 2)
        score += req_pts

    my_preferred_skills = parsed_jd["preferred_skills"]
    matched_pref, pref_pts = [], 0
    if my_preferred_skills:
        matched_pref = [s for s in my_preferred_skills if s.lower() in candidate_skills]
        pref_pts = round((len(matched_pref) / len(my_preferred_skills)) * weights["preferred_skills"], 2)
        score += pref_pts

    req_exp = parsed_jd["experience"]
    my_cand_exp = int(row.get("experience", 0))
    exp_tier, exp_pts = "miss", 0
    if req_exp > 0:
        if my_cand_exp >= req_exp:
            exp_pts = weights["experience"]; exp_tier = "full"
        elif my_cand_exp >= req_exp - 1:
            exp_pts = round(weights["experience"] * 0.7, 2); exp_tier = "partial"
    score += exp_pts

    STOP = {"and", "or", "of", "the", "a", "an", "in", "at", "for", "to"}
    target_kws = [w for w in parsed_jd["role"].lower().split() if w not in STOP]
    cand_role = row.get("current_role", "").lower()
    role_overlap, role_pts = 0.0, 0
    if target_kws:
        overlap = sum(1 for kw in target_kws if kw in cand_role)
        role_overlap = overlap / len(target_kws)
        role_pts = round(role_overlap * weights["role"], 2)
        score += role_pts

    target_loc = parsed_jd["location"].lower().strip()
    my_cand_loc = row.get("location", "").lower().strip()
    if not target_loc:           loc_pts = 3
    elif target_loc == my_cand_loc: loc_pts = weights["location"]
    else:                        loc_pts = 2
    score += loc_pts

    return {
        "match_score":       round(score, 2),
        "req_pts":           req_pts,
        "pref_pts":          pref_pts,
        "exp_pts":           exp_pts,
        "role_pts":          role_pts,
        "loc_pts":           loc_pts,
        "matched_required":  matched_req,
        "matched_preferred": matched_pref,
        "missing_required":  [s for s in req if s not in matched_req] if req else [],
        "exp_tier":          exp_tier,
        "role_overlap_pct":  round(role_overlap * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LIVE CHAT — Recruiter message suggestions (2 per turn, context-aware)
# ─────────────────────────────────────────────────────────────────────────────
def generate_recruiter_suggestions(
    parsed_jd: dict,
    candidate_row: pd.Series,
    conversation_history: list,
    turn_number: int
) -> list:
    """
    Generate 2 smart, context-aware recruiter message suggestions for the current turn.
    Tailored to candidate profile, JD, conversation history, and turn number.
    Returns a list of 2 suggestion strings.
    """
    history_text = ""
    for t in conversation_history:
        history_text += f"Recruiter: {t['recruiter']}\n{candidate_row['name']}: {t['candidate']}\n\n"

    turn_guidance = {
        1: "This is the OPENING message. Introduce yourself, mention the role briefly, and end with ONE question asking if they are open to a conversation. Be warm and specific.",
        2: "This is a FOLLOW-UP message. The candidate has replied. Acknowledge their response and dig deeper — ask about relevant skills, recent projects, or what they are looking for next.",
        3: "This is the CLOSING message. Wrap up the conversation — ask about notice period, availability, or what would make them seriously consider this role. Make it feel like a natural conclusion.",
    }.get(turn_number, "Ask a relevant, natural follow-up question based on the conversation.")

    prompt = f"""You are an expert recruiter coach helping a recruiter craft outreach messages.

Candidate profile:
- Name: {candidate_row['name']}
- Current role: {candidate_row['current_role']}
- Skills: {candidate_row['skills']}
- Experience: {candidate_row['experience']} years
- Location: {candidate_row['location']}

Job being pitched:
- Role: {parsed_jd.get('role', '')}
- Location: {parsed_jd.get('location', '')}
- Required skills: {', '.join(parsed_jd.get('required_skills', []))}
- Preferred skills: {', '.join(parsed_jd.get('preferred_skills', []))}
- Min experience: {parsed_jd.get('experience', 0)}+ years

Conversation so far:
{history_text if history_text else '(No messages yet — this is the first turn.)'}

Turn guidance: {turn_guidance}

Generate exactly 2 recruiter message suggestions. Requirements:
- Each message: 2-3 sentences max, natural and human, not salesy
- Suggestion A: direct and confident tone
- Suggestion B: warm and curious tone
- Both must reference specific details from the candidate profile or JD
- Neither should repeat anything already said in the conversation above

Return ONLY valid JSON — no markdown, no explanation:
{{"suggestions": ["suggestion A here", "suggestion B here"]}}"""

    try:
        text = call_nvidia(prompt, max_tokens=400)
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        suggestions = data.get("suggestions", [])
        while len(suggestions) < 2:
            suggestions.append("")
        return suggestions[:2]
    except Exception:
        # Fallback static suggestions if LLM call fails
        skill1 = candidate_row['skills'].split(',')[0].strip()
        role   = parsed_jd.get('role', 'this role')
        loc    = parsed_jd.get('location', 'our office')
        name   = candidate_row['name']
        fallbacks = {
            1: [
                f"Hi {name}, your {skill1} background is exactly what we're looking for in our {role} opening. Would you be open to a quick 15-minute call to learn more?",
                f"Hey {name}, I came across your profile and your experience as a {candidate_row['current_role']} really stood out. We have a {role} role in {loc} that I think could be a great fit — are you currently exploring opportunities?"
            ],
            2: [
                f"Thanks for sharing that! Given your {skill1} experience, I'm curious — how much of your current work involves {', '.join(parsed_jd.get('required_skills', [skill1])[:2])}? This role is heavily focused on those areas.",
                f"That's really interesting! What's been the most challenging technical problem you've tackled recently, and what are you hoping to find in your next role?"
            ],
            3: [
                f"I appreciate your time, {name}! If things look good after a deeper conversation, what's your current notice period and how soon could you see yourself making a move?",
                f"This has been a great conversation — you seem like a strong fit. Is there anything specific about the role or team you'd want to know before deciding if it's worth exploring further?"
            ]
        }
        return fallbacks.get(turn_number, fallbacks[2])


# ─────────────────────────────────────────────────────────────────────────────
# LIVE CHAT — Candidate reply generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_candidate_reply_live(
    parsed_jd: dict,
    candidate_row: pd.Series,
    conversation_history: list,
    recruiter_message: str
) -> str:
    """
    Generate a realistic, non-deterministic candidate reply to the recruiter's message.
    Uses the full conversation history for context.
    I have set the temperature to 0.85 since we need different reply each run even for identical inputs.
    """
    history_text = ""
    for turn in conversation_history:
        history_text += f"Recruiter: {turn['recruiter']}\n{candidate_row['name']}: {turn['candidate']}\n\n"

    prompt = f"""You are roleplaying as {candidate_row['name']}, a {candidate_row['current_role']} with {candidate_row['experience']} years of experience in the tech industry.

Your profile:
- Skills: {candidate_row['skills']}
- Location: {candidate_row['location']}
- Current role: {candidate_row['current_role']}
- Experience: {candidate_row['experience']} years

The role being discussed:
- Role: {parsed_jd.get('role', 'Software Engineer')}
- Location: {parsed_jd.get('location', 'Not specified')}
- Required skills: {', '.join(parsed_jd.get('required_skills', []))}
- Preferred skills: {', '.join(parsed_jd.get('preferred_skills', []))}
- Minimum experience: {parsed_jd.get('experience', 0)}+ years

Previous conversation:
{history_text if history_text else '(This is the first message from the recruiter.)'}

Recruiter just said: "{recruiter_message}"

Instructions for your reply:
- Reply naturally and authentically as {candidate_row['name']} in 2-4 sentences
- Be REALISTIC: if this role closely matches your skills and experience, show genuine interest and enthusiasm
- If the role is a weak fit (e.g. wrong tech stack, location mismatch), politely show hesitation but remain professional
- Vary your personality — be warm but not overly enthusiastic, ask a natural follow-up question if relevant
- Do NOT copy phrases from previous turns
- Do NOT start with greetings like "Hi" or "Hello" after the first message
- Sound like a real professional, not a script
- Your answer should feel spontaneous and human"""

    return call_nvidia(prompt, max_tokens=200)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Interest Score
# ─────────────────────────────────────────────────────────────────────────────
def score_interest(parsed_jd: dict, row: pd.Series, conversation: list) -> dict:
    full_chat = ""
    for i, turn in enumerate(conversation, 1):
        full_chat += f"[Turn {i}]\nRecruiter: {turn['recruiter']}\n{row['name']}: {turn['candidate']}\n\n"

    prompt = f"""You are an expert recruiter evaluating candidate interest from a real conversation.
Candidate: {row['name']} | {row['current_role']} | {row['experience']} yrs | {row['location']}
Job: {parsed_jd.get('role')} in {parsed_jd.get('location')}

Full conversation:
{full_chat}

Score genuine interest across 4 dimensions (each 0–25):
1. openness       — how actively are they looking / open to opportunities?
2. role_alignment — does this specific role excite them?
3. location_fit   — comfortable with the location/setup?
4. availability   — how soon could they realistically join?

Also extract:
- key_quote: most revealing sentence from the candidate (max 20 words, exact words)
- summary: one sentence overall interest assessment

Return ONLY valid JSON, no markdown:
{{
  "openness": 0, "role_alignment": 0, "location_fit": 0, "availability": 0,
  "key_quote": "", "summary": ""
}}"""
    try:
        text = call_nvidia(prompt, max_tokens=300)
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        def clamp(v): return max(0, min(25, int(v)))
        o = clamp(data.get("openness", 0))
        r = clamp(data.get("role_alignment", 0))
        l = clamp(data.get("location_fit", 0))
        a = clamp(data.get("availability", 0))

        return {
            "interest_score": o + r + l + a,
            "openness": o, "role_alignment": r,
            "location_fit": l, "availability": a,
            "key_quote": data.get("key_quote", ""),
            "summary": data.get("summary", ""),
        }
    except Exception as e:
        return {
            "interest_score": 0, "openness": 0, "role_alignment": 0,
            "location_fit": 0, "availability": 0,
            "key_quote": "", "summary": f"Scoring error: {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def score_badge(s):
    return "🟢" if s >= 75 else "🟡" if s >= 55 else "🟠" if s >= 35 else "🔴"

def interest_label(s):
    return "Highly Interested" if s >= 75 else "Interested" if s >= 55 else "Lukewarm" if s >= 35 else "Low Interest"

def render_chat_bubbles(conversation: list, candidate_name: str):
    html = '<div class="chat-wrap">'
    for i, turn in enumerate(conversation, 1):
        html += f"""
        <div class="bubble-row-recruiter">
            <div class="avatar">🤝</div>
            <div>
                <div class="bubble-label">You (Recruiter) · Turn {i}</div>
                <div class="bubble bubble-recruiter">{turn['recruiter']}</div>
            </div>
        </div>"""
        html += f"""
        <div class="bubble-row-candidate">
            <div>
                <div class="bubble-label bubble-label-right">{candidate_name}</div>
                <div class="bubble bubble-candidate">{turn['candidate']}</div>
            </div>
            <div class="avatar">👤</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def get_chat_state_key(candidate_name: str) -> str:
    """Unique session state key per candidate"""
    return f"chat_{candidate_name.lower().replace(' ', '_')}"


def get_scored_key(candidate_name: str) -> str:
    return f"scored_{candidate_name.lower().replace(' ', '_')}"


def get_suggestion_key(candidate_name: str, turn: int) -> str:
    return f"suggestions_{candidate_name.lower().replace(' ', '_')}_t{turn}"

def get_interest_key(candidate_name: str) -> str:
    return f"interest_{candidate_name.lower().replace(' ', '_')}"



MAX_RECRUITER_TURNS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────
init_db()

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "parsed_jd" not in st.session_state:
    st.session_state.parsed_jd = None
if "all_match_results" not in st.session_state:
    st.session_state.all_match_results = []
if "top_candidates" not in st.session_state:
    st.session_state.top_candidates = []
if "final_results" not in st.session_state:
    st.session_state.final_results = []
if "scouting_done" not in st.session_state:
    st.session_state.scouting_done = False
if "active_chat_candidate" not in st.session_state:
    st.session_state.active_chat_candidate = None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎯 Talent Scout AI")

    st.subheader("➕ Add Candidate")
    with st.form("add_candidate_form", clear_on_submit=True):
        new_name     = st.text_input("Full Name",                placeholder="e.g. Rohan Sharma")
        new_role     = st.text_input("Current Role",             placeholder="e.g. Backend Developer")
        new_skills   = st.text_input("Skills (comma separated)", placeholder="e.g. Python, FastAPI, AWS")
        new_exp      = st.number_input("Years of Experience", min_value=0, max_value=40, value=3, step=1)
        new_location = st.text_input("Location",                 placeholder="e.g. Bangalore")
        submitted    = st.form_submit_button("Add Candidate", use_container_width=True)
        if submitted:
            if new_name and new_role and new_skills and new_location:
                add_candidate(new_name, new_skills, int(new_exp), new_location, new_role)
                st.success(f"✅ {new_name} added!")
                st.rerun()
            else:
                st.warning("Please fill in all fields.")

    st.divider()

    st.subheader("👥 Candidate Pool")
    df_sidebar = get_candidates()
    if df_sidebar.empty:
        st.info("No candidates yet.")
    else:
        for _, crow in df_sidebar.iterrows():
            col_info, col_del = st.columns([4, 1])
            with col_info:
                st.markdown(
                    f"**{crow['name']}** · {crow['current_role']}\n\n"
                    f"<span style='font-size:12px;color:gray'>{crow['location']} · {crow['experience']} yrs</span>",
                    unsafe_allow_html=True
                )
            with col_del:
                if st.button("🗑️", key=f"del_{crow['id']}", help=f"Delete {crow['name']}"):
                    delete_candidate(int(crow["id"]))
                    st.rerun()

    st.divider()

    st.subheader("🔑 API Key Pool")
    if not _ALL_KEYS:
        st.error("No API keys found. Add NVIDIA_API_KEY_1 to your .env")
    else:
        for i, key in enumerate(_ALL_KEYS):
            label = f"Key {i+1}: ...{key[-6:]}"
            if i == _key_state["index"]:
                st.success(f"🟢 {label} — active")
            else:
                st.info(f"⚪ {label} — standby")
        st.caption(
            f"{len(_ALL_KEYS)} key(s) · 40 rpm each · "
            f"2s gap enforced · max {len(_ALL_KEYS) * 30} rpm combined"
        )

    st.divider()

    st.subheader("⚙️ Score Weights")
    match_weight    = st.slider("Match weight", 0.4, 0.8, 0.6, 0.05)
    interest_weight = round(1 - match_weight, 2)
    st.caption(f"Final = Match × {match_weight} + Interest × {interest_weight}")

    if st.session_state.scouting_done:
        st.divider()
        if st.button("🔄 New Search", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎯 Talent Scout AI")
st.subheader("AI-Powered Talent Scouting & Engagement Agent")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — JD Input & Scouting (only shown before scouting)
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.scouting_done:
    jd = st.text_area(
        "📋 Paste Job Description Here", height=260,
        placeholder="We are hiring a Python Backend Developer with 3+ years of experience in FastAPI and AWS..."
    )

    run = st.button("🚀 Scout Candidates", type="primary", use_container_width=True)

    if run:
        if not jd.strip():
            st.warning("Please paste a Job Description first.")
            st.stop()
        if not _ALL_KEYS:
            st.error("No NVIDIA API keys found in .env file.")
            st.stop()

        df = get_candidates()
        if df.empty:
            st.warning("No candidates in pool. Add some using the sidebar.")
            st.stop()

        _key_state["index"] = 0
        _last_call_time.clear()

        with st.spinner("🤖 Parsing Job Description..."):
            parsed_jd = parse_jd(jd)
        st.session_state.parsed_jd = parsed_jd

        with st.spinner(f"⚡ Scoring all {len(df)} candidates by match (no API calls)..."):
            all_match_results = []
            for _, row in df.iterrows():
                match_data = calculate_match_score(parsed_jd, row)
                all_match_results.append((row, match_data))
            all_match_results.sort(key=lambda x: x[1]["match_score"], reverse=True)

        st.session_state.all_match_results = all_match_results
        st.session_state.top_candidates = all_match_results[:3]
        st.session_state.scouting_done = True
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Live Chat & Results (after scouting)
# ─────────────────────────────────────────────────────────────────────────────
else:
    parsed_jd        = st.session_state.parsed_jd
    all_match_results = st.session_state.all_match_results
    top_candidates   = st.session_state.top_candidates

    # ── Parsed JD summary ────────────────────────────────────────────────────
    with st.expander("📄 Parsed Job Description", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Role",     parsed_jd["role"] or "—")
        c2.metric("Min Exp",  f"{parsed_jd['experience']}+ yrs")
        c3.metric("Location", parsed_jd["location"] or "Not specified")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Required Skills**")
            for s in parsed_jd["required_skills"]: st.markdown(f"- `{s}`")
        with cb:
            st.markdown("**Preferred Skills**")
            for s in parsed_jd["preferred_skills"]: st.markdown(f"- `{s}`")

    # ── Full pool scores ──────────────────────────────────────────────────────
    with st.expander(f"📊 Full Pool — Match Scores (all {len(all_match_results)} candidates)", expanded=False):
        preview_rows = []
        for rank, (row, md) in enumerate(all_match_results, 1):
            preview_rows.append({
                "Rank":       rank,
                "":           score_badge(md["match_score"]),
                "Name":       row["name"],
                "Role":       row["current_role"],
                "Match /100": md["match_score"],
                "Req Skills": f"{len(md['matched_required'])}/{len(parsed_jd['required_skills'])}",
                "Exp Tier":   md["exp_tier"],
                "Location":   row["location"],
            })
        st.dataframe(
            pd.DataFrame(preview_rows), use_container_width=True, hide_index=True,
            column_config={"Match /100": st.column_config.ProgressColumn(
                "Match", min_value=0, max_value=100, format="%.1f"
            )}
        )

    top_names = [row["name"] for row, _ in top_candidates]
    st.success(f"🏅 Top 3 candidates selected for live outreach: **{', '.join(top_names)}**")

    # ────────────────────────────────────────────────────────────────────────
    # LIVE CHAT SECTION
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💬 Live Recruiter Outreach")
    st.markdown(
        "Chat with each candidate (up to **3 messages**). "
        "The AI will respond as the candidate based on their real profile. "
        "After your 3rd message, interest scoring runs automatically."
    )

    all_chats_complete = True

    for cand_idx, (cand_row, match_data) in enumerate(top_candidates):
        name       = cand_row["name"]
        chat_key   = get_chat_state_key(name)
        scored_key = get_scored_key(name)
        int_key    = get_interest_key(name)

        # Initialise chat history for this candidate
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
        if scored_key not in st.session_state:
            st.session_state[scored_key] = False
        if int_key not in st.session_state:
            st.session_state[int_key] = None

        conversation     = st.session_state[chat_key]
        already_scored   = st.session_state[scored_key]
        turns_used       = len(conversation)
        turns_left       = MAX_RECRUITER_TURNS - turns_used
        chat_complete    = already_scored

        if not chat_complete:
            all_chats_complete = False

        # ── Candidate card header ─────────────────────────────────────────
        badge = score_badge(match_data["match_score"])
        tier_icon = {"full": "✅", "partial": "🟡", "miss": "❌"}.get(match_data["exp_tier"], "")

        if chat_complete:
            interest_data = st.session_state[int_key]
            final = round(
                (match_data["match_score"] * match_weight) +
                (interest_data["interest_score"] * interest_weight), 2
            )
            expander_label = (
                f"{badge} **#{cand_idx+1} {name}** — "
                f"Match: {match_data['match_score']} | Interest: {interest_data['interest_score']} | "
                f"Final: **{final}**"
            )
        else:
            expander_label = (
                f"{badge} **#{cand_idx+1} {name}** — "
                f"Match: {match_data['match_score']} | "
                f"{'✅ Chat complete' if already_scored else f'💬 {turns_left} message(s) remaining'}"
            )

        with st.expander(expander_label, expanded=(not chat_complete)):

            # ── Match breakdown (compact) ─────────────────────────────────
            with st.container():
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Req Skills", f"{match_data['req_pts']:.0f}/45")
                m2.metric("Pref Skills", f"{match_data['pref_pts']:.0f}/15")
                m3.metric("Experience", f"{match_data['exp_pts']:.0f}/20",
                          delta=f"{tier_icon} {match_data['exp_tier']}", delta_color="off")
                m4.metric("Role", f"{match_data['role_pts']:.0f}/15")
                m5.metric("Location", f"{match_data['loc_pts']:.0f}/5")

                col_a, col_b = st.columns(2)
                with col_a:
                    if match_data["matched_required"]:
                        st.success(f"✅ Matched: {', '.join(match_data['matched_required'])}")
                    if match_data["missing_required"]:
                        st.error(f"❌ Missing: {', '.join(match_data['missing_required'])}")
                with col_b:
                    if match_data["matched_preferred"]:
                        st.info(f"⭐ Preferred: {', '.join(match_data['matched_preferred'])}")

            st.markdown("---")

            # ── Chat history ──────────────────────────────────────────────
            if conversation:
                render_chat_bubbles(conversation, name)

            # ── Live input (only if turns remain and not yet scored) ──────
            if not already_scored:
                if turns_left > 0:
                    # Turn counter badge
                    if turns_left == 1:
                        badge_cls = "turn-counter-warn"
                        badge_text = f"⚠️ Last message to {name}"
                    else:
                        badge_cls = "turn-counter"
                        badge_text = f"💬 {turns_left} message(s) left"

                    st.markdown(
                        f'<span class="{badge_cls}">{badge_text}</span>',
                        unsafe_allow_html=True
                    )

                    # Just AI Suggestions 
                    sugg_key    = get_suggestion_key(name, turns_used + 1)
                    prefill_key = f"prefill_{name}_{turns_used}"

                    # Generate suggestions once per turn (cache in session state)
                    if sugg_key not in st.session_state:
                        with st.spinner(f"💡 Generating smart suggestions for Turn {turns_used + 1}..."):
                            try:
                                suggestions = generate_recruiter_suggestions(
                                    parsed_jd, cand_row, conversation, turns_used + 1
                                )
                            except RuntimeError:
                                suggestions = ["", ""]
                        st.session_state[sugg_key] = suggestions

                    suggestions = st.session_state[sugg_key]

                    st.markdown(
                        '<div class="suggestion-label">💡 AI Suggestions — click to use, or type your own below</div>',
                        unsafe_allow_html=True
                    )
                    s_col1, s_col2 = st.columns(2)
                    with s_col1:
                        if suggestions[0]:
                            if st.button(
                                f"Option A: {suggestions[0][:60]}{'…' if len(suggestions[0]) > 60 else ''}",
                                key=f"sugg_a_{name}_{turns_used}",
                                use_container_width=True,
                                help=suggestions[0]
                            ):
                                st.session_state[prefill_key] = suggestions[0]
                                st.rerun()
                    with s_col2:
                        if suggestions[1]:
                            if st.button(
                                f"Option B: {suggestions[1][:60]}{'…' if len(suggestions[1]) > 60 else ''}",
                                key=f"sugg_b_{name}_{turns_used}",
                                use_container_width=True,
                                help=suggestions[1]
                            ):
                                st.session_state[prefill_key] = suggestions[1]
                                st.rerun()

                    st.markdown(
                        '<p class="hint-text">✏️ Hover over a suggestion to see the full text. Click to auto-fill, then edit freely before sending.</p>',
                        unsafe_allow_html=True
                    )

                    # Text area — pre-filled if a suggestion was clicked
                    # Write directly into the widget's own session state key so Streamlit picks it up
                    input_key = f"input_{name}_{turns_used}"
                    if prefill_key in st.session_state:
                        st.session_state[input_key] = st.session_state.pop(prefill_key)

                    recruiter_input = st.text_area(
                        f"Your message to {name}",
                        key=input_key,
                        height=110,
                        placeholder=f"Hi {name}, I came across your profile and wanted to reach out about an exciting opportunity..."
                    )

                    send_col, refresh_col, _ = st.columns([1, 1, 2])
                    with send_col:
                        send_btn = st.button(
                            f"Send →  (Turn {turns_used + 1}/{MAX_RECRUITER_TURNS})",
                            key=f"send_{name}_{turns_used}",
                            type="primary"
                        )
                    with refresh_col:
                        if st.button("🔄 New suggestions", key=f"refresh_sugg_{name}_{turns_used}"):
                            # Clear cached suggestions to regenerate
                            if sugg_key in st.session_state:
                                del st.session_state[sugg_key]
                            st.rerun()

                    if send_btn:
                        if not recruiter_input.strip():
                            st.warning("Please type a message before sending.")
                        else:
                            with st.spinner(f"🤔 {name} is composing a reply..."):
                                try:
                                    candidate_reply = generate_candidate_reply_live(
                                        parsed_jd, cand_row, conversation, recruiter_input.strip()
                                    )
                                except RuntimeError as e:
                                    st.error(f"API error: {e}")
                                    st.stop()

                            # Append turn
                            st.session_state[chat_key].append({
                                "recruiter": recruiter_input.strip(),
                                "candidate": candidate_reply
                            })

                            # If this was the last turn, auto-score
                            letsc_new_turns = len(st.session_state[chat_key])
                            if letsc_new_turns >= MAX_RECRUITER_TURNS:
                                with st.spinner(f"📊 Analysing {name}'s interest level..."):
                                    try:
                                        interest_data = score_interest(
                                            parsed_jd, cand_row, st.session_state[chat_key]
                                        )
                                    except RuntimeError as e:
                                        st.error(f"Scoring error: {e}")
                                        interest_data = {
                                            "interest_score": 0, "openness": 0,
                                            "role_alignment": 0, "location_fit": 0,
                                            "availability": 0, "key_quote": "",
                                            "summary": "Could not score — API error."
                                        }
                                st.session_state[int_key]    = interest_data
                                st.session_state[scored_key] = True

                            st.rerun()

                else:
                    # Turns exhausted but score not yet triggered (edge case guard)
                    with st.spinner(f"📊 Analysing {name}'s interest level..."):
                        interest_data = score_interest(parsed_jd, cand_row, conversation)
                    st.session_state[int_key]    = interest_data
                    st.session_state[scored_key] = True
                    st.rerun()

            # ── Interest results (After all the scoring) 
            if already_scored and st.session_state[int_key]:
                my_interest_data = st.session_state[int_key]
                st.markdown("---")
                st.markdown("#### 💡 Interest Score Breakdown")
                i1, i2, i3, i4, i5 = st.columns(5)
                i1.metric("Total", f"{my_interest_data['interest_score']} / 100")
                i2.metric("Openness",       f"{my_interest_data['openness']} / 25")
                i3.metric("Role Alignment", f"{my_interest_data['role_alignment']} / 25")
                i4.metric("Location Fit",   f"{my_interest_data['location_fit']} / 25")
                i5.metric("Availability",   f"{my_interest_data['availability']} / 25")
                st.markdown(f"**AI Assessment:** {my_interest_data['summary']}")
                if my_interest_data["key_quote"]:
                    st.info(f'💬 *"{my_interest_data["key_quote"]}"*')

                final = round(
                    (match_data["match_score"] * match_weight) +
                    (my_interest_data["interest_score"] * interest_weight), 2
                )
                st.success(
                    f"**Combined Final Score: {final} / 100** "
                    f"(Match {match_data['match_score']} × {match_weight} + "
                    f"Interest {interest_data['interest_score']} × {interest_weight})"
                )

    # ────────────────────────────────────────────────────────────────────────
    # FINAL RANKED SHORTLIST (shown after all 3 chats complete)
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("---")

    # Count how many are done
    my_done_chats = sum(
        1 for row, _ in top_candidates
        if st.session_state.get(get_scored_key(row["name"]), False)
    )

    if my_done_chats < len(top_candidates):
        remaining = len(top_candidates) - my_done_chats
        st.info(
            f"💬 Complete outreach with **{remaining} more candidate(s)** to unlock the final ranked shortlist."
        )
    else:
        # Build final results
        ranked = []
        for cand_row, match_data in top_candidates:
            name          = cand_row["name"]
            my_interest_data = st.session_state[get_interest_key(name)]
            final         = round(
                (match_data["match_score"] * match_weight) +
                (my_interest_data["interest_score"] * interest_weight), 2
            )
            ranked.append({
                "name": name, "current_role": cand_row["current_role"],
                "skills": cand_row["skills"], "experience": cand_row["experience"],
                "location": cand_row["location"],
                "match_score": match_data["match_score"],
                "interest_score": my_interest_data["interest_score"],
                "final_score": final,
                "req_pts": match_data["req_pts"], "pref_pts": match_data["pref_pts"],
                "exp_pts": match_data["exp_pts"], "role_pts": match_data["role_pts"],
                "loc_pts": match_data["loc_pts"],
                "openness": my_interest_data["openness"],
                "role_alignment": my_interest_data["role_alignment"],
                "location_fit": my_interest_data["location_fit"],
                "availability": my_interest_data["availability"],
                "interest_summary": my_interest_data["summary"],
                "key_quote": my_interest_data["key_quote"],
            })

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        st.session_state.final_results = ranked

        st.subheader("🏆 Final Ranked Shortlist")
        st.success(f"Top picks: **{', '.join(r['name'] for r in ranked)}**")

        table_rows = [{
            "Rank": i + 1,
            "":     score_badge(r["final_score"]),
            "Name": r["name"],
            "Current Role": r["current_role"],
            "Match /100":    r["match_score"],
            "Interest /100": r["interest_score"],
            "Final /100":    r["final_score"],
            "Interest Level": interest_label(r["interest_score"]),
        } for i, r in enumerate(ranked)]

        st.dataframe(
            pd.DataFrame(table_rows), use_container_width=True, hide_index=True,
            column_config={
                "Match /100":    st.column_config.ProgressColumn("Match",    min_value=0, max_value=100, format="%.1f"),
                "Interest /100": st.column_config.ProgressColumn("Interest", min_value=0, max_value=100, format="%.1f"),
                "Final /100":    st.column_config.ProgressColumn("Final",    min_value=0, max_value=100, format="%.1f"),
            }
        )

        # CSV export
        st.divider()
        export_cols = [
            "name", "current_role", "skills", "experience", "location",
            "match_score", "interest_score", "final_score",
            "req_pts", "pref_pts", "exp_pts", "role_pts", "loc_pts",
            "openness", "role_alignment", "location_fit", "availability",
            "interest_summary", "key_quote"
        ]
        csv = pd.DataFrame(ranked)[export_cols].to_csv(index=False)
        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv, file_name="talent_scout_results.csv", mime="text/csv",
            use_container_width=True
        )