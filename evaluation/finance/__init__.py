"""In-house evaluation for the finance research agent: citation verification + data accuracy.

Layered design: deterministic rules first (reproducible, zero cost); an LLM judge only for what rules cannot do (semantic entailment);
and a meta-evaluation that injects known errors to measure the evaluator's own detection and false-positive rates.
"""
