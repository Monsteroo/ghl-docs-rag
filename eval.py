import json
from typing import Callable

from retrieval import retrieve


def score_retrieval(questions: list[dict], retrieve_fn: Callable) -> dict:
    correct = 0
    misses = []
    for q in questions:
        results = retrieve_fn(q["question"], top_k=5)
        titles = [r["title"] for r in results]
        if any(q["expected_title_contains"] in t for t in titles):
            correct += 1
        else:
            misses.append({"question": q["question"], "expected_title_contains": q["expected_title_contains"]})
    return {
        "total": len(questions),
        "correct": correct,
        "accuracy": correct / len(questions) if questions else 0.0,
        "misses": misses,
    }


def run_eval() -> None:
    with open("eval_questions.json") as f:
        questions = json.load(f)
    result = score_retrieval(questions, retrieve)
    print(f"Retrieval accuracy: {result['correct']}/{result['total']} ({result['accuracy']:.0%})")
    if result["misses"]:
        print("Missed:")
        for miss in result["misses"]:
            print(f"  - {miss['question']} (expected title containing: {miss['expected_title_contains']})")


if __name__ == "__main__":
    run_eval()
