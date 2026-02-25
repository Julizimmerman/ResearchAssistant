"""Agent implementations for the research assistant pipeline.

Four agents:
- SupervisorAgent  — orchestrates the pipeline, no LLM calls
- InvestigatorAgent — decomposes a topic into subtopics
- CuratorAgent      — deep analysis per subtopic
- ReporterAgent     — synthesises analyses into a Markdown report
"""
