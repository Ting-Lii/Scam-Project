"""
Chain of Thought (CoT) prompt templates for Gemini AI analysis of employment/job scams
Based on academic research (Ravenelle et al., 2022) and FTC guidelines

This module provides prompts that use Chain of Thought reasoning to guide the model
through step-by-step analysis before providing final conclusions.
"""

import json
from pathlib import Path
from functools import lru_cache

# Load classification framework once at module level
_FRAMEWORK_PATH = Path(__file__).parent / "scam_classification_framework.json"
_FRAMEWORK = None
_FRAMEWORK_SUMMARY = None  # Cache the summary string

def _load_framework():
    """Load the classification framework from JSON file (cached)"""
    global _FRAMEWORK
    if _FRAMEWORK is None:
        try:
            with open(_FRAMEWORK_PATH, 'r') as f:
                _FRAMEWORK = json.load(f)
        except FileNotFoundError:
            # Fallback to empty framework if file not found
            _FRAMEWORK = {}
    return _FRAMEWORK

def _compute_framework_summary():
    """Compute the framework summary (internal function)"""
    framework = _load_framework()
    if not framework:
        return "Use standard job scam classification (Ravenelle et al., 2022)."
    
    # Build summary with red flag categories and their items
    summary_parts = []
    
    if "scam_categories" in framework:
        categories = list(framework["scam_categories"].keys())
        summary_parts.append(f"Scam Categories: {', '.join(categories)}")
    
    if "red_flag_categories" in framework:
        red_flags = framework["red_flag_categories"]
        summary_parts.append("Red Flag Categories:")
        for category, items in red_flags.items():
            if isinstance(items, dict):
                # Dictionary format: show category and item count
                item_list = list(items.values())[:3]  # Show first 3 as examples
                examples = "; ".join(item_list)
                summary_parts.append(f"  - {category}: {examples}... ({len(items)} items)")
            elif isinstance(items, list):
                # Legacy array format
                examples = "; ".join(items[:3])
                summary_parts.append(f"  - {category}: {examples}... ({len(items)} items)")
    
    if "vulnerability_factors" in framework:
        vuln_factors = framework["vulnerability_factors"]
        if isinstance(vuln_factors, dict):
            examples = "; ".join(list(vuln_factors.values())[:3])
            summary_parts.append(f"Vulnerability Factors: {examples}... ({len(vuln_factors)} items)")
        elif isinstance(vuln_factors, list):
            examples = "; ".join(vuln_factors[:3])
            summary_parts.append(f"Vulnerability Factors: {examples}... ({len(vuln_factors)} items)")
    
    return "\n".join(summary_parts) if summary_parts else ""

@lru_cache(maxsize=1)
def _get_framework_summary():
    """Get a minimal condensed summary of the framework for the prompt (cached)"""
    return _compute_framework_summary()

# Pre-compute and cache the framework summary at module load time for instant access
_FRAMEWORK_SUMMARY = _compute_framework_summary()

SCAM_ANALYSIS_PROMPT_COT_TEMPLATE = """Analyze this job complaint for scam indicators using Chain of Thought reasoning.

Framework Reference:
{framework_summary}

IMPORTANT: Use the exact red_flag categories from the framework:
- "communication": Communication-related red flags (unsolicited contact, poor grammar, pressure tactics, etc.)
- "financial": Financial red flags (payment requests, bank info requests, fake checks, etc.)
- "job_posting": Job posting red flags (unrealistic pay, vague descriptions, suspicious postings, etc.)
- "hiring_process": Hiring process red flags (no interview, immediate hiring, document requests, etc.)
- "work_activity": Work activity red flags (money mule tasks, package reshipping, payment processing, etc.)

CHAIN OF THOUGHT ANALYSIS - Follow these steps:

STEP 1: Initial Reading and Context Understanding
- Read the complaint carefully and identify the main situation described
- Note any key entities mentioned (company names, job titles, contact methods)
- Identify the timeline of events if described
- Note the complainant's perspective and concerns

STEP 2: Systematic Red Flag Identification
For each red flag category, analyze the complaint systematically:

a) Communication Analysis:
   - How was the initial contact made? (email, text, phone, job board)
   - What is the quality of communication? (grammar, professionalism, clarity)
   - Are there pressure tactics or urgency signals?
   - Is the communication channel appropriate for a legitimate job?

b) Financial Analysis:
   - Are there any requests for money, payment, or financial information?
   - When in the process were financial requests made?
   - What type of financial information or transactions are involved?
   - Are payment methods suspicious or unusual?

c) Job Posting Analysis:
   - What are the job details? (title, description, requirements, pay)
   - Are the job details realistic for the position?
   - Is there sufficient company information provided?
   - Are there "too good to be true" elements?

d) Hiring Process Analysis:
   - What was the hiring process? (application, interview, background check)
   - How quickly did the process move?
   - Were standard hiring practices followed?
   - Were any documents requested, and when?

e) Work Activity Analysis:
   - What tasks or activities were described or requested?
   - Do the tasks seem legitimate for the job type?
   - Are there any suspicious activities (money movement, package handling, etc.)?

STEP 3: Pattern Recognition and Scam Type Classification
- Review all identified red flags together
- Look for patterns that match known scam types from the framework
- Consider which primary category and subcategory best fit the evidence
- Assess the strength of evidence (Strong/Moderate/Weak)

STEP 4: Risk Assessment
- Calculate scam_probability (0-100) based on:
  * Number and severity of red flags identified
  * Alignment with known scam patterns
  * Evidence strength
  * Absence of legitimate indicators
- Assess financial_risk level considering:
  * Potential financial loss mentioned or implied
  * Type of financial requests made
  * Urgency and pressure tactics
- Evaluate victim_profile vulnerability factors:
  * What factors made the victim susceptible?
  * What is the overall risk level for this victim profile?

STEP 5: Synthesis and Final Assessment
- Combine all findings into a coherent assessment
- Determine confidence level (0-100) based on:
  * Clarity of evidence
  * Consistency of red flags
  * Alignment with known patterns
- Generate actionable recommendations based on the analysis

STEP 6: Output Generation
Now provide your final analysis in JSON format:

Return JSON:
{{
    "reasoning_steps": {{
        "step1_context": "<brief summary of the situation>",
        "step2_red_flags": {{
            "communication": ["<flag1>", "<flag2>"],
            "financial": ["<flag1>", "<flag2>"],
            "job_posting": ["<flag1>", "<flag2>"],
            "hiring_process": ["<flag1>", "<flag2>"],
            "work_activity": ["<flag1>", "<flag2>"]
        }},
        "step3_pattern_analysis": "<explanation of patterns found and scam type reasoning>",
        "step4_risk_calculation": "<explanation of how scam_probability and risk levels were determined>",
        "step5_confidence": "<explanation of confidence level>"
    }},
    "scam_probability": <0-100>,
    "scam_type": {{"primary_category": "<name>", "subcategory": "<type>"}},
    "red_flags": {{"communication": [], "financial": [], "job_posting": [], "hiring_process": [], "work_activity": []}},
    "financial_risk": {{"level": "<Low/Medium/High/Critical>", "potential_loss": "<amount>", "explanation": "<brief>"}},
    "victim_profile": {{"vulnerability_factors": [], "risk_level": "<Low/Medium/High>"}},
    "evidence_strength": "<Strong/Moderate/Weak>",
    "recommendations": {{"immediate_actions": [], "verification_steps": [], "reporting": []}},
    "confidence": <0-100>
}}

Complaint:
{text}"""


def get_scam_analysis_prompt_cot(text: str) -> str:
    """
    Generate the Chain of Thought prompt for employment scam analysis
    
    Args:
        text: The complaint text to analyze
        
    Returns:
        Formatted prompt string with Chain of Thought reasoning steps
    """
    # Use pre-computed cached summary for instant access
    framework_summary = _FRAMEWORK_SUMMARY if _FRAMEWORK_SUMMARY is not None else _get_framework_summary()
    return SCAM_ANALYSIS_PROMPT_COT_TEMPLATE.format(
        text=text,
        framework_summary=framework_summary
    )

