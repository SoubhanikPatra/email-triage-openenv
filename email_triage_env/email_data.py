"""
Synthetic B2B SaaS email corpus with gold-standard triage labels.

Each entry is a dict with:
  subject, body, sender_email, sender_name,
  thread_length, has_attachment, account_tier, prior_tickets,
  gold_priority, gold_category, gold_routing, gold_sentiment,
  gold_requires_followup, gold_tags, difficulty
"""

EMAILS = [
    # -------------------------------------------------------------------------
    # EASY — clear signals, unambiguous routing
    # -------------------------------------------------------------------------
    {
        "email_id": "e001",
        "subject": "Invoice #4821 — payment failed",
        "body": (
            "Hi team,\n\n"
            "We tried to process our monthly subscription renewal this morning "
            "but the payment failed. Our finance team says the card details on "
            "file are outdated. Could you please send a new payment link?\n\n"
            "Best,\nDavid Chen\nAcme Corp"
        ),
        "sender_email": "david.chen@acmecorp.com",
        "sender_name": "David Chen",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "pro",
        "prior_tickets": 1,
        "gold_priority": "high",
        "gold_category": "billing",
        "gold_routing": "billing_team",
        "gold_sentiment": "neutral",
        "gold_requires_followup": True,
        "gold_tags": ["invoice", "payment", "renewal"],
        "difficulty": "easy",
    },
    {
        "email_id": "e002",
        "subject": "Can't log in — password reset not working",
        "body": (
            "Hello,\n\n"
            "I've tried resetting my password three times today and the reset "
            "email never arrives. I've checked my spam folder. This is blocking "
            "my entire team from working.\n\n"
            "Regards,\nSarah Kim"
        ),
        "sender_email": "sarah.kim@startupxyz.io",
        "sender_name": "Sarah Kim",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "starter",
        "prior_tickets": 0,
        "gold_priority": "high",
        "gold_category": "technical_support",
        "gold_routing": "tier1_support",
        "gold_sentiment": "negative",
        "gold_requires_followup": False,
        "gold_tags": ["login", "password", "reset", "blocked"],
        "difficulty": "easy",
    },
    {
        "email_id": "e003",
        "subject": "Would love to see dark mode!",
        "body": (
            "Hey,\n\n"
            "Love the product. One thing that would make it even better: dark mode. "
            "Any chance this is on the roadmap?\n\nThanks!"
        ),
        "sender_email": "user@freelancer.net",
        "sender_name": "Alex Rivera",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "free",
        "prior_tickets": 0,
        "gold_priority": "low",
        "gold_category": "feature_request",
        "gold_routing": "tier1_support",
        "gold_sentiment": "positive",
        "gold_requires_followup": False,
        "gold_tags": ["dark mode", "feature", "ui"],
        "difficulty": "easy",
    },
    {
        "email_id": "e004",
        "subject": "Unsubscribe me NOW",
        "body": "STOP EMAILING ME. I never signed up for this.",
        "sender_email": "angry@yahoo.com",
        "sender_name": "Unknown",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "free",
        "prior_tickets": 0,
        "gold_priority": "normal",
        "gold_category": "account_management",
        "gold_routing": "tier1_support",
        "gold_sentiment": "very_negative",
        "gold_requires_followup": False,
        "gold_tags": ["unsubscribe", "opt-out"],
        "difficulty": "easy",
    },
    {
        "email_id": "e005",
        "subject": "Interested in enterprise pricing",
        "body": (
            "Hi,\n\n"
            "We're a 500-person company evaluating tools. Could someone walk us "
            "through enterprise pricing and SSO options?\n\nBest,\nMike"
        ),
        "sender_email": "mike.t@bigcorp.com",
        "sender_name": "Mike Thompson",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "free",
        "prior_tickets": 0,
        "gold_priority": "high",
        "gold_category": "sales_inquiry",
        "gold_routing": "sales_team",
        "gold_sentiment": "positive",
        "gold_requires_followup": True,
        "gold_tags": ["enterprise", "pricing", "sso", "sales"],
        "difficulty": "easy",
    },

    # -------------------------------------------------------------------------
    # MEDIUM — mixed signals, requires contextual reasoning
    # -------------------------------------------------------------------------
    {
        "email_id": "m001",
        "subject": "Re: Re: Re: Ongoing sync issues",
        "body": (
            "This is the fourth time I'm writing about the same issue. "
            "Our data hasn't synced properly since the update two weeks ago. "
            "We're an enterprise customer paying $3,000/month and getting worse "
            "service than before. If this isn't fixed by Friday I'm cancelling.\n\n"
            "— Janet"
        ),
        "sender_email": "janet.w@globalfirm.com",
        "sender_name": "Janet Wu",
        "thread_length": 3,
        "has_attachment": True,
        "account_tier": "enterprise",
        "prior_tickets": 4,
        "gold_priority": "urgent",
        "gold_category": "technical_support",
        "gold_routing": "tier2_support",
        "gold_sentiment": "very_negative",
        "gold_requires_followup": True,
        "gold_tags": ["sync", "data", "enterprise", "escalation", "churn-risk"],
        "difficulty": "medium",
    },
    {
        "email_id": "m002",
        "subject": "Weird behavior when exporting to PDF",
        "body": (
            "Hi support,\n\n"
            "When I export a report with more than 50 rows to PDF, the last page "
            "is always cut off. Happens in Chrome and Firefox. Running v3.4.1.\n\n"
            "Steps to reproduce:\n"
            "1. Create report with 51+ rows\n"
            "2. Click Export > PDF\n"
            "3. Last page truncated\n\n"
            "Let me know if you need logs."
        ),
        "sender_email": "pete@midsize.co",
        "sender_name": "Pete Alvarez",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "pro",
        "prior_tickets": 2,
        "gold_priority": "normal",
        "gold_category": "bug_report",
        "gold_routing": "tier2_support",
        "gold_sentiment": "neutral",
        "gold_requires_followup": True,
        "gold_tags": ["pdf", "export", "bug", "report", "truncated"],
        "difficulty": "medium",
    },
    {
        "email_id": "m003",
        "subject": "Billing question + upgrade inquiry",
        "body": (
            "Hello,\n\n"
            "Two things:\n\n"
            "1. Our last invoice shows a charge for 15 seats but we only have 12 "
            "active users. Can you fix this?\n\n"
            "2. We're growing and might need 30 seats by Q3. What would that cost?\n\n"
            "Thanks,\nLena"
        ),
        "sender_email": "lena.m@growthco.com",
        "sender_name": "Lena Marsh",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "pro",
        "prior_tickets": 1,
        "gold_priority": "high",
        "gold_category": "billing",
        "gold_routing": "billing_team",
        "gold_sentiment": "neutral",
        "gold_requires_followup": True,
        "gold_tags": ["billing", "seats", "upgrade", "invoice"],
        "difficulty": "medium",
    },
    {
        "email_id": "m004",
        "subject": "API rate limits — help needed",
        "body": (
            "Hi,\n\n"
            "We're hitting 429 errors on the /reports/generate endpoint during "
            "peak hours (8–10 AM EST). Our integration is batching 200 requests "
            "per minute. Is there a way to get higher limits or should we queue?\n\n"
            "We're on the Pro plan."
        ),
        "sender_email": "dev@techstartup.io",
        "sender_name": "Dev Team",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "pro",
        "prior_tickets": 0,
        "gold_priority": "normal",
        "gold_category": "technical_support",
        "gold_routing": "tier2_support",
        "gold_sentiment": "neutral",
        "gold_requires_followup": False,
        "gold_tags": ["api", "rate-limit", "429", "integration"],
        "difficulty": "medium",
    },
    {
        "email_id": "m005",
        "subject": "Request: SAML SSO configuration guide",
        "body": (
            "Hello,\n\n"
            "We're trying to set up SAML SSO with Okta. Our IT admin followed "
            "the docs but is getting an 'Invalid assertion' error. Do you have "
            "a step-by-step guide for Okta specifically, or can someone jump on "
            "a call?\n\nThanks,\nTom"
        ),
        "sender_email": "tom.r@enterprise-customer.com",
        "sender_name": "Tom Richards",
        "thread_length": 1,
        "has_attachment": False,
        "account_tier": "enterprise",
        "prior_tickets": 0,
        "gold_priority": "high",
        "gold_category": "technical_support",
        "gold_routing": "tier2_support",
        "gold_sentiment": "neutral",
        "gold_requires_followup": True,
        "gold_tags": ["sso", "saml", "okta", "authentication"],
        "difficulty": "medium",
    },

    # -------------------------------------------------------------------------
    # HARD — ambiguous, multi-layered, requires nuanced judgment
    # -------------------------------------------------------------------------
    {
        "email_id": "h001",
        "subject": "A few thoughts",
        "body": (
            "Hi there,\n\n"
            "Long-time user here. I wanted to share some feedback.\n\n"
            "First, the new dashboard is really slick — our team loves it. "
            "However, we noticed something strange: when we run the nightly batch "
            "jobs, the system sometimes marks completed tasks as 'pending'. It "
            "doesn't happen every night, only when two jobs run within 30 seconds "
            "of each other. Could be a race condition?\n\n"
            "Also, we're about to bring on 20 more team members. Is there a "
            "volume discount for Pro?\n\n"
            "Separately, could you make the CSV export include the 'created_by' "
            "field? It's missing and we need it for compliance auditing.\n\n"
            "Cheers,\nNaomi"
        ),
        "sender_email": "naomi.k@regulated-industry.com",
        "sender_name": "Naomi Keane",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "pro",
        "prior_tickets": 3,
        "gold_priority": "high",
        "gold_category": "bug_report",
        "gold_routing": "tier2_support",
        "gold_sentiment": "positive",
        "gold_requires_followup": True,
        "gold_tags": ["race-condition", "batch", "compliance", "csv", "upgrade"],
        "difficulty": "hard",
    },
    {
        "email_id": "h002",
        "subject": "Fwd: Fwd: Urgent — client demo tomorrow",
        "body": (
            "--- Forwarded message ---\n\n"
            "Hi, forwarding this from our VP. We have a demo for a Fortune 500 "
            "prospect tomorrow at 9 AM. The custom branding feature we were "
            "promised in March still isn't working — logo doesn't appear in "
            "exported reports. Our sales rep (James) said it was fixed but it's "
            "not. We need this resolved tonight or we'll lose a $2M deal. "
            "Calling in every favour here.\n\n"
            "— Rachel (CTO, OurBigClient Inc.)"
        ),
        "sender_email": "rachel.t@bigclient.com",
        "sender_name": "Rachel Torres",
        "thread_length": 2,
        "has_attachment": False,
        "account_tier": "enterprise",
        "prior_tickets": 7,
        "gold_priority": "urgent",
        "gold_category": "bug_report",
        "gold_routing": "escalation",
        "gold_sentiment": "very_negative",
        "gold_requires_followup": True,
        "gold_tags": ["demo", "branding", "escalation", "enterprise", "revenue-risk"],
        "difficulty": "hard",
    },
    {
        "email_id": "h003",
        "subject": "Data export question",
        "body": (
            "Hello,\n\n"
            "We need to export all user activity logs for the past 18 months "
            "for a legal audit. We couldn't find this in the UI. Is this "
            "available via API? We need it within 48 hours.\n\n"
            "Please treat this as confidential.\n\nRegards,\nGareth"
        ),
        "sender_email": "gareth.p@financial-services.com",
        "sender_name": "Gareth Price",
        "thread_length": 0,
        "has_attachment": False,
        "account_tier": "enterprise",
        "prior_tickets": 1,
        "gold_priority": "urgent",
        "gold_category": "account_management",
        "gold_routing": "escalation",
        "gold_sentiment": "neutral",
        "gold_requires_followup": True,
        "gold_tags": ["data-export", "legal", "audit", "compliance", "confidential"],
        "difficulty": "hard",
    },
    {
        "email_id": "h004",
        "subject": "Re: Renewal — not happy",
        "body": (
            "I got the auto-renewal notice but no one told me the price was "
            "increasing by 40%. I'm also seeing charges I don't recognise. "
            "We've had three outages this quarter. At this point I'm seriously "
            "considering moving to a competitor. I'd like to speak to someone "
            "senior to discuss our options — including a refund for downtime.\n\n"
            "— Frank"
        ),
        "sender_email": "frank.b@midmarket.com",
        "sender_name": "Frank Bauer",
        "thread_length": 1,
        "has_attachment": False,
        "account_tier": "pro",
        "prior_tickets": 5,
        "gold_priority": "urgent",
        "gold_category": "billing",
        "gold_routing": "escalation",
        "gold_sentiment": "very_negative",
        "gold_requires_followup": True,
        "gold_tags": ["renewal", "price-increase", "churn-risk", "refund", "outage"],
        "difficulty": "hard",
    },
    {
        "email_id": "h005",
        "subject": "Integration with Salesforce — strange data",
        "body": (
            "Hi,\n\n"
            "Our Salesforce integration has been running for 6 months. Recently "
            "we noticed some contacts are being duplicated when a deal stage "
            "changes from 'Proposal' to 'Negotiation'. This only started after "
            "your v4.2 release. We have 12,000 contacts so a manual fix isn't "
            "feasible.\n\n"
            "I've attached a CSV of the affected records. Could your team write "
            "a migration script, or should we do it ourselves via API?\n\n"
            "Also — unrelated — our contract renews in 60 days. Can you send "
            "renewal terms?\n\nThanks,\nSophia"
        ),
        "sender_email": "sophia.l@salesdriven.com",
        "sender_name": "Sophia Lim",
        "thread_length": 0,
        "has_attachment": True,
        "account_tier": "enterprise",
        "prior_tickets": 2,
        "gold_priority": "high",
        "gold_category": "bug_report",
        "gold_routing": "tier2_support",
        "gold_sentiment": "neutral",
        "gold_requires_followup": True,
        "gold_tags": ["salesforce", "integration", "duplicate", "migration", "renewal"],
        "difficulty": "hard",
    },
]


# Convenience groupings
EASY_EMAILS = [e for e in EMAILS if e["difficulty"] == "easy"]
MEDIUM_EMAILS = [e for e in EMAILS if e["difficulty"] == "medium"]
HARD_EMAILS = [e for e in EMAILS if e["difficulty"] == "hard"]

TASK_EMAIL_MAP = {
    "easy_triage": EASY_EMAILS,
    "medium_triage": MEDIUM_EMAILS,
    "hard_triage": HARD_EMAILS,
}