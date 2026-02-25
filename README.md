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

## Key Design Decisions

- **Explicit SupervisorAgent**: `SupervisorAgent` owns the full pipeline lifecycle — initialising agents, building the graph, and managing the stream → interrupt → resume cycle. The CLI only handles user I/O and delegates all orchestration to the Supervisor.
- **graph.py as graph definition only**: `graph.py` defines the LangGraph `StateGraph` topology (nodes, edges, `build_graph()`). Execution logic lives in the Supervisor, not in the graph module.
- **`interrupt()`-based Human-in-the-Loop**: LangGraph's native `interrupt()` pauses the graph at the human review node, checkpoints state to `MemorySaver`, and resumes via `Command(resume=...)`. The Supervisor mediates this cycle; the CLI contributes only the UI callback.
- **Structured Outputs**: All inter-agent communication uses Pydantic v2 models with `with_structured_output()`, ensuring type-safe, validated data at every boundary.
- **Complexity-Aware Routing**: `ModelRouter` maps each agent's complexity level to the appropriate Azure OpenAI deployment — cheap model for simple tasks, powerful model for reasoning-heavy ones.
- **Parallel Curation**: `CuratorAgent` uses `ThreadPoolExecutor` to analyse all approved subtopics concurrently, minimising wall-clock time.
- **Retry with backoff**: `BaseAgent._call_structured_llm` wraps every LLM call with Tenacity (3 attempts, exponential backoff) to handle transient API errors transparently.

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

## Project Structure

```
research_assistant/
├── __init__.py          Package metadata
├── __main__.py          python -m entry point
├── cli.py               CLI: argument parsing, user I/O, delegates to Supervisor
├── config.py            Settings loaded from .env (pydantic-settings)
├── models.py            All Pydantic v2 data models (inter-agent contracts)
├── state.py             LangGraph ResearchState TypedDict
├── graph.py             StateGraph definition: nodes, edges, build_graph()
├── routing.py           Complexity-aware model router (cheap vs. powerful)
├── ui.py                Rich console UI: panels, tables, spinners, Markdown
├── parser.py            Human review command parser
├── mock.py              Mock LLM for --mock mode
└── agents/
    ├── __init__.py
    ├── base.py          Abstract base agent: structured LLM calls + retry
    ├── supervisor.py    Supervisor: pipeline orchestration, HITL cycle
    ├── investigator.py  Subtopic generation (gpt-4o-mini)
    ├── curator.py       Deep analysis, parallel execution (gpt-4o)
    └── reporter.py      Report synthesis (gpt-4o)
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
