# Research Assistant

A console-based multi-agent system that researches any topic, validates findings with human input, and produces a polished Markdown report — all while optimising AI costs by routing tasks to models of appropriate capability.

## Architecture

```
User enters topic
    → Investigator Agent (gpt-4o-mini)
        Decomposes the topic into 5-7 research subtopics
    → Human Review (console)
        User approves, rejects, modifies, or adds subtopics
    → Curator Agent (gpt-4o)
        Performs deep analysis on each approved subtopic
    → Reporter Agent (gpt-4o)
        Synthesises everything into a structured Markdown report
    → Output
        Report saved to file + cost summary displayed
```

### The Four Agents

| Agent | Responsibility | Model | Why |
|---|---|---|---|
| **Investigator** | Topic decomposition into subtopics | `gpt-4o-mini` | Simple task — cheap model suffices |
| **Curator** | Deep analysis per subtopic (key findings, pros/cons, connections) | `gpt-4o` | Complex reasoning requires a powerful model |
| **Reporter** | Final Markdown report synthesis | `gpt-4o` | High-quality writing needs the best model |
| **Supervisor** | Pipeline orchestration | None | Implemented as the LangGraph state graph topology itself |

### Key Design Decisions

- **Graph-as-Supervisor**: The LangGraph `StateGraph` edge topology IS the supervisor. Since the workflow is linear, a separate supervisor node would add complexity without value. The graph structure declaratively encodes the orchestration.
- **`interrupt()`-based Human-in-the-Loop**: Uses LangGraph's native `interrupt()` mechanism to pause the graph, checkpoint state, and surface subtopics to the CLI. This cleanly separates agent logic from UI concerns and enables crash recovery.
- **Structured Outputs**: All inter-agent communication uses Pydantic v2 models with `with_structured_output()`, ensuring type-safe, validated data flows.
- **Cost-Aware Routing**: A `ModelRouter` maps agent complexity levels to appropriate models. Simple tasks use cheap models; complex tasks use powerful ones.
- **SQLite Response Cache**: LangChain's `SQLiteCache` eliminates redundant API calls for identical prompts, reducing cost during development and repeated research.

## Setup

### Prerequisites

- Python 3.11+
- An Azure OpenAI resource with `gpt-4o-mini` and `gpt-4o` deployments (or use `--mock` mode for testing)

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
python -m research_assistant "Artificial Intelligence in Healthcare"
```

### Mock Mode (No API Key Required)

```bash
python -m research_assistant "Artificial Intelligence" --mock
```

This runs the entire pipeline with realistic fake LLM responses — useful for testing and for reviewers without an API key.

### All Options

```bash
python -m research_assistant "topic" [options]

Options:
  --mock              Run with mock LLM responses (no API calls)
  --verbose           Enable debug logging
  --max-subtopics N   Maximum subtopics (default: 7)
  -o, --output PATH   Custom output file path
  --version           Show version
```

### Human Review Commands

When the Investigator presents subtopics, you can use these commands:

```
approve 1,3,5          Approve specific subtopics by ID
approve all            Approve all subtopics
reject 2,4             Reject specific subtopics
add 'AI Safety'        Add a new subtopic
modify 3 to 'New Name' Rename a subtopic
done                   Finish review and proceed
```

Commands can be combined with `;` on a single line:
```
approve 1,3,5; reject 2; add 'Ethics'; done
```

## Project Structure

```
research_assistant/
├── __init__.py          Package metadata
├── __main__.py          python -m entry point
├── cli.py               CLI argument parsing and execution loop
├── config.py            Settings from .env (pydantic-settings)
├── models.py            All Pydantic v2 data models
├── state.py             LangGraph TypedDict state definition
├── graph.py             StateGraph construction, nodes, agent registry
├── routing.py           Cost-aware model router
├── cost.py              LLM usage and cost tracking
├── cache.py             SQLite LLM response cache
├── ui.py                Rich console UI (panels, tables, spinners)
├── parser.py            Human-in-the-loop command parser
├── mock.py              Mock LLM for --mock mode
└── agents/
    ├── __init__.py
    ├── base.py          Abstract base agent with retry + cost tracking
    ├── investigator.py  Subtopic generation (cheap model)
    ├── curator.py       Deep analysis (expensive model)
    └── reporter.py      Report synthesis (expensive model)
```

## Technical Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — State graph orchestration with `interrupt()` for human-in-the-loop
- **[LangChain](https://github.com/langchain-ai/langchain)** + **langchain-openai** — Azure OpenAI integration with structured outputs
- **[Pydantic v2](https://docs.pydantic.dev/)** — Typed data models for all inter-agent communication
- **[Rich](https://github.com/Textualize/rich)** — Polished console UI with panels, tables, spinners, and Markdown rendering
- **[Tenacity](https://github.com/jd/tenacity)** — Retry with exponential backoff on LLM calls
- **[python-dotenv](https://github.com/theskumar/python-dotenv)** — Environment variable management
