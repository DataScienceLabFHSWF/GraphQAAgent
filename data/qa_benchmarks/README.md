# QA Benchmarks

Gold-standard question-answer pairs for evaluating the KG-RAG agent.

## Format

Each benchmark file is a JSON array of objects:

```json
{
  "question_id": "q01",
  "question": "Natural language question",
  "expected_answer": "Expected answer text",
  "expected_entities": ["Entity1", "Entity2"],
  "difficulty": "easy|medium|hard",
  "question_type": "factoid|list|boolean|comparative|causal|aggregation",
  "competency_question_id": "CQ-01"  // optional link to CQ
}
```
