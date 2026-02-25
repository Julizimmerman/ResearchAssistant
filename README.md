# Research Assistant

A console-based multi-agent system that researches any topic, validates findings with human input, and produces a polished Markdown report — displayed directly in the terminal.

## Architecture

```
User enters topic
    → Supervisor Agent
        Initialises agents, builds the graph, orchestrates the full pipeline
        ↓
    → Investigator Agent (gpt-4o-mini)
        Decomposes the topic into up to 7 focused subtopics
        ↓
    → Human Review (console interrupt)
        User approves, rejects, modifies, or adds subtopics
        ↓
    → Curator Agent (gpt-4o)
        Performs deep analysis on each approved subtopic in parallel
        ↓
    → Reporter Agent (gpt-4o)
        Synthesises all analyses into a structured Markdown report
        ↓
    → Output
        Report rendered in the terminal
```

## The Four Agents

| Agent | File | Responsibility | Model |
|---|---|---|---|
| **Supervisor** | `agents/supervisor.py` | Orchestrates the pipeline, manages the stream → interrupt → resume cycle, logs agent handoffs | None (no LLM calls) |
| **Investigator** | `agents/investigator.py` | Decomposes a topic into subtopics with relevance scores | `gpt-4o-mini` |
| **Curator** | `agents/curator.py` | Deep analysis per subtopic — key findings, pros/cons, connections, gaps. Runs subtopics in parallel | `gpt-4o` |
| **Reporter** | `agents/reporter.py` | Synthesises curated analyses into a polished original Markdown report | `gpt-4o` |


## Setup

### Prerequisites

- Python 3.10+
- An Azure OpenAI resource with `gpt-4o-mini` and `gpt-4o` deployments — or use `--mock` mode for testing without credentials

### Installation

```bash
# Clone the repository
git clone https://github.com/Julizimmerman/ResearchAssistant.git
cd ResearchAssistant

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your Azure OpenAI credentials
```

## Usage

### Basic Usage

```bash
python -m research_assistant
```

### Mock Mode (No API Key Required)

```bash
python -m research_assistant --mock
```

Runs the entire pipeline with realistic pre-built LLM responses — useful for testing and for reviewers without an Azure API key.

### All Options

```
Options:
  --mock              Run with mock LLM responses (no API calls)
  --verbose           Enable debug logging (shows agent handoffs and graph events)
  --max-subtopics N   Maximum subtopics for the Investigator (default: 7)
  --version           Show version
```

### Human Review Commands

After the Investigator runs, the Supervisor pauses for human input. Available commands:

```
approve 1,3,5          Approve specific subtopics by ID
approve all            Approve all subtopics
reject 2,4             Reject specific subtopics
add 'Topic Name'       Add a new subtopic
modify 3 to 'New Name' Rename a subtopic
done                   Finish review and proceed
```

Multiple commands can be combined with `;` on one line:

```
approve 1,3; reject 2; add 'Ethics'; done
```

## Data Flow

Each agent communicates through typed Pydantic models:

```
InvestigatorAgent  →  list[Subtopic]
                   ↓
        Human review → HumanDecision
                   ↓
CuratorAgent       →  list[CuratedAnalysis]
                   ↓
ReporterAgent      →  FinalReport
```

All models are defined in `models.py` and also serve as the JSON schemas sent to the LLM via `with_structured_output()`.

## Technical Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — State graph orchestration with `interrupt()` for human-in-the-loop and `MemorySaver` for checkpointing
- **[LangChain](https://github.com/langchain-ai/langchain)** + **langchain-openai** — Azure OpenAI integration with structured outputs
- **[Pydantic v2](https://docs.pydantic.dev/)** — Typed data models for all inter-agent communication
- **[Rich](https://github.com/Textualize/rich)** — Polished console UI with panels, tables, spinners, and Markdown rendering
- **[Tenacity](https://github.com/jd/tenacity)** — Retry with exponential backoff on LLM calls
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** — Environment variable management
