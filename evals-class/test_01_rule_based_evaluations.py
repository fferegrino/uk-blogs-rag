import json
import pytest
from common import get_response

with open("test_contains_cases.json", "r") as f:
    test_cases = json.load(f)


def check_expected_words(response, expected_words):
    for word in expected_words:
        assert (
            word.lower() in response.lower()
        ), f"The word {word} is not in the response"


def no_lists_in_response(response):
    assert "1. " not in response, "The response contains a list"


@pytest.mark.parametrize("test_case", test_cases["tests"], ids=test_cases["test_names"])
def test_prompt(test_case):
    article = test_case["article"]
    expected_words = test_case["expected_words"]

    instruction = (
        "Give me a short summary of the following article, "
        "maximum a couple of sentences with no lists or quotes:"
    )

    full_prompt = f"{instruction}\n\n{article}"

    response = get_response(full_prompt)

    with open(f"test_model_response_{test_case['id']}.txt", "w") as f:
        f.write(response)

    check_expected_words(response, expected_words)
    no_lists_in_response(response)
