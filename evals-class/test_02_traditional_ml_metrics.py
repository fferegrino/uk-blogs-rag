import json
import pytest
import csv
from common import get_response

ACCURACY_THRESHOLD = 0.8


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def test_accuracy():
    instruction = """Classify the following article into one of the following categories:
    - AI
    - music
    - food
    - education

    If the article is not about any of the categories above, classify it as "unknown".

    Provide no explanation, or any other text, just the category.

    This is the article:

    {article}
    """

    with open("test_classification_cases.json", "r") as f:
        test_cases = json.load(f)

    y_true = [test_case["category"] for test_case in test_cases["tests"]]
    y_pred = []

    for test_case in test_cases["tests"]:
        full_prompt = instruction.format(article=test_case["article"])
        y_pred.append(get_response(full_prompt))

    with open("test_model_response_classification.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "y_true", "y_pred", "article_excerpt"])
        for y_true, y_pred, test_case in zip(y_true, y_pred, test_cases["tests"]):
            writer.writerow(
                [test_case["id"], y_true, y_pred, test_case["article"][:100]]
            )

    accuracy = accuracy_score(y_true, y_pred)
    assert (
        accuracy >= ACCURACY_THRESHOLD
    ), f"Accuracy is {accuracy}, expected {ACCURACY_THRESHOLD}"
