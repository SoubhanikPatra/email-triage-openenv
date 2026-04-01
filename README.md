# Email Triage Environment

> **OpenEnv-compatible RL environment** for training and evaluating AI agents on realistic B2B SaaS email triage.

[![HF Space](https://img.shields.io/badge/🤗-HF%20Space-yellow)](https://huggingface.co/spaces)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

---

## What is this?

Support inbox triage is one of the most common, high-value tasks in any B2B SaaS company. Every day, customer success teams must:

- **Prioritise** urgency (urgent → low)
- **Categorise** the issue (billing, bug, feature request, …)
- **Route** to the right team (billing team, tier-2 engineering, escalation, …)
- **Detect sentiment** to handle churn-risk customers carefully
- **Tag** emails for tracking and analytics

This environment simulates a realistic support inbox. An agent reads emails one at a time and makes a structured triage decision. A programmatic grader scores each decision against gold-standard labels — giving rich, dense reward signal even on partial correctness.

---

## Quick Start

### 1. Run locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### 2. Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### 3. Deploy to HF Spaces

```bash
openenv push --repo-id your-username/email-triage-env
```

### 4. Run the baseline inference script

```bash
export HF_TOKEN=hf_...
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

---

## Environment API

All endpoints are plain HTTP — no WebSocket required.

| Endpoint        | Method | Description                        |
| --------------- | ------ | ---------------------------------- |
| `/health`       | GET    | Health check                       |
| `/tasks`        | GET    | List available tasks with metadata |
| `/reset`        | POST   | Start a new episode                |
| `/step`         | POST   | Submit a triage action             |
| `/state`        | GET    | Get episode-level metadata         |
| `/session/{id}` | DELETE | Clean up session                   |

### Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy_triage"}'
```

```json
{
  "session_id": "abc-123",
  "observation": {
    "email_id": "e002",
    "subject": "Can't log in — password reset not working",
    "body": "...",
    "sender_email": "sarah.kim@startupxyz.io",
    "account_tier": "starter",
    "prior_tickets": 0,
    "step_number": 0,
    "total_emails": 5,
    "done": false,
    "reward": null
  }
}
```

### Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123",
    "action": {
      "priority": "high",
      "category": "technical_support",
      "routing_target": "tier1_support",
      "sentiment": "negative",
      "requires_followup": false,
      "summary": "User cannot log in; password reset emails not arriving.",
      "suggested_response_tone": "empathetic",
      "tags": ["login", "password", "reset"]
    }
  }'
```

```json
{
  "session_id": "abc-123",
  "reward": 0.95,
  "done": false,
  "observation": {
    "feedback": "Issues: Routing: predicted 'tier1_support' but gold is 'tier1_support'",
    "score_breakdown": {
      "priority": 1.0,
      "category": 1.0,
      "routing": 1.0,
      "sentiment": 1.0,
      "followup": 1.0,
      "tag_overlap": 0.5
    },
    ...
  }
}
```

---

## Observation Space

| Field             | Type | Description                                   |
| ----------------- | ---- | --------------------------------------------- |
| `email_id`        | str  | Unique email identifier                       |
| `subject`         | str  | Email subject line                            |
| `body`            | str  | Full email body                               |
| `sender_email`    | str  | Sender's email address                        |
| `sender_name`     | str  | Sender's display name                         |
| `thread_length`   | int  | Number of prior emails in thread              |
| `has_attachment`  | bool | Whether email has attachments                 |
| `account_tier`    | str  | `free \| starter \| pro \| enterprise`        |
| `prior_tickets`   | int  | Number of open tickets from this sender       |
| `step_number`     | int  | Current step within the episode               |
| `total_emails`    | int  | Total emails in this episode                  |
| `feedback`        | str  | Human-readable grader feedback on last action |
| `score_breakdown` | dict | Per-dimension scores from last step           |

## Action Space

| Field                     | Type      | Valid values                                                                                                                      |
| ------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `priority`                | str       | `urgent \| high \| normal \| low`                                                                                                 |
| `category`                | str       | `billing \| technical_support \| feature_request \| bug_report \| account_management \| sales_inquiry \| general_inquiry \| spam` |
| `routing_target`          | str       | `tier1_support \| tier2_support \| billing_team \| sales_team \| engineering \| account_management \| spam_filter \| escalation`  |
| `sentiment`               | str       | `positive \| neutral \| negative \| very_negative`                                                                                |
| `requires_followup`       | bool      | Whether a follow-up action is needed within 24h                                                                                   |
| `summary`                 | str       | ≤ 2-sentence summary for the support rep                                                                                          |
| `suggested_response_tone` | str       | `formal \| friendly \| empathetic \| concise`                                                                                     |
| `tags`                    | list[str] | 0–5 keyword tags                                                                                                                  |

---

## Reward Function

Each step returns a reward in **[0.0, 1.0]** computed as a weighted sum:

| Dimension   | Weight | Scoring                                       |
| ----------- | ------ | --------------------------------------------- |
| Priority    | 0.25   | Exact = 1.0, Adjacent tier = 0.5, Wrong = 0.0 |
| Category    | 0.25   | Exact = 1.0, else 0.0                         |
| Routing     | 0.20   | Exact = 1.0, else 0.0                         |
| Sentiment   | 0.10   | Exact = 1.0, Adjacent = 0.5, else 0.0         |
| Follow-up   | 0.10   | Exact bool match                              |
| Tag overlap | 0.10   | Jaccard similarity (partial credit)           |

**Partial credit design:** Predicting `high` instead of `urgent` scores 0.5 on priority, not 0.0 — the agent is rewarded for being in the right ballpark.

---

## Tasks

### `easy_triage` (difficulty: easy)

5 emails with clear, unambiguous signals:

- Payment failure → billing team
- Password reset broken → tier1 support
- Feature request for dark mode → tier1 support
- Unsubscribe request → account management
- Enterprise pricing inquiry → sales team

Expected baseline score: ~0.75–0.85 for GPT-class models.

### `medium_triage` (difficulty: medium)

5 emails requiring context:

- Enterprise customer with repeat sync issues (churn risk → tier2, not tier1)
- PDF export bug with clear steps to reproduce
- Combined billing discrepancy + upgrade inquiry
- API rate limiting question from developer
- SAML SSO setup failure for enterprise customer

Expected baseline score: ~0.60–0.75.

### `hard_triage` (difficulty: hard)

5 emails with competing signals, legal implications, or multi-issue content:

- One email covering a bug, a pricing question, AND a feature request
- Forwarded CTO email about a $2M demo at risk — needs escalation, not tier2
- Legal audit data export request (confidential, 48h deadline)
- Churn-risk renewal dispute with outage refund request
- Salesforce integration bug + contract renewal in same email

Expected baseline score: ~0.45–0.65.

---

## Project Structure

```
email-triage-env/
├── email_triage_env/
│   ├── __init__.py
│   ├── models.py          # Typed Pydantic models (Action, Observation, State)
│   ├── email_data.py      # 15 synthetic emails with gold labels
│   └── grader.py          # Deterministic multi-dimension grader
├── server/
│   ├── __init__.py
│   ├── environment.py     # reset() / step() / state logic
│   └── app.py             # FastAPI application
├── inference.py           # Baseline inference script (OpenAI client)
├── openenv.yaml           # OpenEnv manifest
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Baseline Results

Scores from `meta-llama/Llama-3.1-8B-Instruct` (greedy decoding):

| Task          | Avg reward | Notes                                        |
| ------------- | ---------- | -------------------------------------------- |
| easy_triage   | ~0.80      | Struggles slightly on tag overlap            |
| medium_triage | ~0.65      | Misses escalation signals in long threads    |
| hard_triage   | ~0.52      | Multi-issue emails confuse routing decisions |

---

## License

Apache 2.0
