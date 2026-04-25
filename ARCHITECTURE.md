# Talent Scout AI — Architecture & Scoring Logic

## Overview

Talent Scout AI is a Streamlit web app that helps recruiters quickly turn a job description into a ranked shortlist of candidates.

The app combines:

- Rule-based candidate matching for speed and transparency
- AI-powered conversations for realistic recruiter outreach
- AI-based interest scoring from candidate replies

This helps reduce manual screening time and improves hiring speed.

---

## End-to-End Flow

## Step 1 — Parse the Job Description

The recruiter pastes any job description into the app.

AI extracts:

- Role title
- Required skills
- Preferred skills
- Location
- Minimum experience

Example:

{
  "role": "Python Backend Developer",
  "required_skills": ["Python", "FastAPI", "SQL", "AWS"],
  "preferred_skills": ["Docker"],
  "location": "Bangalore",
  "experience": 3
}

This becomes the benchmark for candidate matching.

---

## Step 2 — Match Score (0 to 100)

Each candidate gets a score using 5 factors:

| Factor | Weight |
|--------|--------|
| Required Skills | 45 |
| Preferred Skills | 15 |
| Experience | 20 |
| Role Alignment | 15 |
| Location Fit | 5 |

### Required Skills

More matched required skills = higher score.

A smoothing curve is used so partially qualified candidates still get fair credit.

### Preferred Skills

Bonus points for optional skills.

### Experience

- Meets requirement = full score
- Slightly below = partial score
- Far below = zero

### Role Alignment

Checks similarity between target role and candidate’s current title.

Example:

Backend Developer ↔ Senior Backend Engineer

### Location Fit

- Same city = full score
- Different city = lower score
- No location given = neutral score

---

## Step 3 — Shortlist Top Candidates

After scoring everyone, top 3 candidates move forward.

This keeps the app fast and reduces API usage.

---

## Step 4 — AI Recruiter Conversation

Recruiters can chat with shortlisted candidates.

The app:

1. Suggests recruiter outreach messages
2. Allows custom messages
3. Generates realistic candidate replies using AI

Example:

Recruiter: Are you open to backend opportunities in Bangalore?

Candidate: Yes, this matches my recent FastAPI experience. I'd like to know more.

---

## Step 5 — Interest Score (0 to 100)

After the chat, AI scores interest using 4 areas:

| Category | Max Score |
|----------|-----------|
| Openness | 25 |
| Role Alignment | 25 |
| Location Fit | 25 |
| Availability | 25 |

Examples:

- Is the candidate open to changing jobs?
- Does the role interest them?
- Are they okay with the location?
- Can they join soon?

This helps find candidates who are both qualified and interested.

---

## Step 6 — Final Ranking

Final score combines both scores:

Final Score = Match Score × 60% + Interest Score × 40%

Recruiters can change this ratio.

Example:

- More Match weight = prioritize skills
- More Interest weight = prioritize engagement

---

## Candidate Data

The app uses SQLite database:

talent.db

Each candidate has:

- Name
- Skills
- Experience
- Location
- Current Role

Recruiters can also add or delete candidates.

---

## AI Model Used

NVIDIA NIM API

Model:

meta/llama-3.3-70b-instruct

Used for:

- JD parsing
- Candidate replies
- Message suggestions
- Interest scoring

---

## Reliability Features

To avoid failures:

- Multiple API keys supported
- Auto key rotation on quota errors
- Retry on temporary failures
- Handles timeout safely

---

## Why This Design Works

Rule-based logic is used for:

- matching
- ranking
- calculations

AI is used for:

- conversations
- intent understanding
- interest scoring

This gives speed, explainability, and smart decision-making.

---

## Final Outcome

Talent Scout AI helps recruiters answer two questions quickly:

1. Who is qualified?
2. Who is genuinely interested?
