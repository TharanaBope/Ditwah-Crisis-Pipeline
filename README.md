# Ditwah-Crisis-Pipeline

**Operation Ditwah** - AI-Powered Disaster Response System for Post-Cyclone Relief

## Overview

A standalone Crisis Intelligence Pipeline built for the **AI Engineer Essentials** bootcamp. This system processes real-time crisis data during disaster scenarios using multi-provider LLM integration, demonstrating practical prompt engineering techniques for life-critical applications.

**Scenario:** Post-Cyclone Ditwah Relief Operations in Sri Lanka (December 2025)

## Features

| Part      | Feature                 | Points  | Description                                         |
| --------- | ----------------------- | ------- | --------------------------------------------------- |
| 1         | Few-Shot Classification | 20      | Classify crisis messages using labeled examples     |
| 2         | Temperature Stress Test | 15      | Demonstrate determinism in crisis systems           |
| 3         | CoT & ToT Logistics     | 20      | Chain-of-Thought scoring + Tree-of-Thought routing  |
| 4         | Token Economics         | 15      | Block/summarize messages exceeding token limits     |
| 5         | News Feed Extraction    | 30      | Structured data extraction with Pydantic validation |
| **Total** |                         | **100** |                                                     |

## Tech Stack

- **LLM Providers:** OpenAI, Google Gemini, Groq
- **Framework:** Python 3.11+
- **Validation:** Pydantic v2
- **Data Processing:** Pandas, OpenPyXL
- **Token Counting:** Tiktoken
- **Configuration:** YAML-driven

## Project Structure

```
Mini-Project-0/
├── mini_project_0.ipynb      # Main notebook (all 5 parts)
├── pyproject.toml            # Project dependencies
├── uv.lock                   # Dependency lock file
├── .env.sample               # API key template
├── .gitignore
├── README.md
│
├── utils/                    # Multi-provider LLM utilities
│   ├── __init__.py
│   ├── llm_client.py         # LLM abstraction (OpenAI, Gemini, Groq)
│   ├── router.py             # Model selection & routing
│   ├── config_loader.py      # YAML config loading
│   ├── token_utils.py        # Token counting
│   ├── prompts.py            # Versioned prompt templates
│   ├── json_utils.py         # JSON parsing & Pydantic helpers
│   └── logging_utils.py      # CSV call logging
│
├── config/                   # Configuration files
│   ├── config.yaml           # Behavior configuration
│   └── models.yaml           # Provider/model tier mappings
│
├── data/                     # Crisis scenario data
│   ├── Sample Messages.txt   # 50 mixed messages for classification
│   ├── Scenarios.txt         # 2 complex crisis scenarios
│   ├── Incidents.txt         # 3 incidents for logistics planning
│   └── News Feed.txt         # 30 news items for extraction
│
├── output/                   # Generated deliverables
│   ├── classified_messages.xlsx
│   └── flood_report.xlsx
│
└── logs/                     # API call logs
    └── runs.csv
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/crisis-intel-pipeline.git
cd Ditwah-Crisis-Pipeline
```

### 2. Create environment and install dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 3. Configure API keys

```bash
cp .env.sample .env
```

Edit `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
GROQ_API_KEY=gsk-...
```

### 4. Run the notebook

```bash
jupyter notebook mini_project_0.ipynb
```

## Usage

### Part 1: Few-Shot Classification

Classifies crisis messages into categories using 8 labeled examples:

```
Input:  "SOS: 5 people trapped on roof in Ja-Ela"
Output: District: Gampaha | Intent: Rescue | Priority: High
```

**Categories:** Rescue, Supply, Info, Other

### Part 2: Temperature Stress Test

Demonstrates why crisis systems require `temperature=0.0`:

- **Chaos Mode (temp=1.0):** Variable outputs, potential hallucinations
- **Safe Mode (temp=0.0):** Deterministic, reliable outputs

### Part 3: CoT & ToT Logistics

**Chain-of-Thought Scoring:**

- Base Score: 5
- +2 if Age > 60 or < 5
- +3 if Need == "Rescue"
- +1 if Need == "Insulin/Medicine"

**Tree-of-Thought Route Planning:**

- Branch 1: Greedy (highest score first)
- Branch 2: Speed (closest first)
- Branch 3: Logistics (furthest first)

### Part 4: Token Economics

Guards against spam/chain messages:

```python
if tokens > 150:
    print("BLOCKED/TRUNCATED")
    # Summarize using overflow_summarize.v1
```

### Part 5: News Feed Extraction

Extracts structured data using Pydantic models:

```python
class CrisisEvent(BaseModel):
    district: str
    flood_level_meters: Optional[float]
    victim_count: int
    main_need: str
    status: Literal["Critical", "Warning", "Stable"]
```

## Results

### Classification Summary (Part 1)

| Intent | Count |
| ------ | ----- |
| Info   | 22    |
| Rescue | 12    |
| Other  | 9     |
| Supply | 7     |

### Crisis Events Extracted (Part 5)

| Status   | Count |
| -------- | ----- |
| Critical | 15    |
| Stable   | 7     |
| Warning  | 6     |

**Total Victims Reported:** 20,829

**Most Affected Districts:** Colombo, Gampaha

## Deliverables

| File                              | Description                            |
| --------------------------------- | -------------------------------------- |
| `output/classified_messages.xlsx` | 50 classified crisis messages          |
| `output/flood_report.xlsx`        | 28 structured crisis events (3 sheets) |

## Key Learnings

1. **Few-shot prompting** dramatically improves classification accuracy
2. **Temperature=0.0** is essential for deterministic crisis systems
3. **Chain-of-Thought** enables transparent decision-making
4. **Tree-of-Thought** explores multiple strategies before deciding
5. **Token guards** protect against API cost overruns
6. **Pydantic validation** ensures data quality in extraction pipelines

## License

This project is part of the AI Engineer Essentials bootcamp curriculum.

## Acknowledgments

- **Zuu Crew** - AI Engineer Essentials Bootcamp
- **Scenario:** Operation Ditwah - Cyclone Relief (Sri Lanka)
