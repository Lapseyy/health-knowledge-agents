# Alternative Evaluation Pattern: DeepEval

RAGAS is the required/reference evaluation surface for Assignment 2. DeepEval can be used as an additional engineering comparison, but it does not replace the required RAGAS report unless the instructor explicitly changes the assignment contract.

A DeepEval experiment can add pytest-style assertions around response quality, retrieval relevance, hallucination, or task-specific criteria. Keep it separate from the canonical `src/eval/ragas_eval.py` path so graders can still reproduce the required metrics.

If included, pin the current DeepEval version and document any evaluator-model costs.
