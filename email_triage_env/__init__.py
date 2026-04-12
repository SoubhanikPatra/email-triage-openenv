"""
Email Triage Environment - Core Package
Provides the RL environment for email triage tasks.
"""

from email_triage_env.models import TriageAction, EmailObservation, TriageState
from email_triage_env.grader import grade, grade_detailed
from email_triage_env.email_data import EMAILS, EASY_EMAILS, MEDIUM_EMAILS, HARD_EMAILS, TASK_EMAIL_MAP

__all__ = [
    "TriageAction",
    "EmailObservation", 
    "TriageState",
    "grade",
    "grade_detailed",
    "EMAILS",
    "EASY_EMAILS", 
    "MEDIUM_EMAILS",
    "HARD_EMAILS",
    "TASK_EMAIL_MAP",
]