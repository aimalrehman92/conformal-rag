import re


def extract_tag_content(input_string):
    # Use regular expression to find all content between <s> and </s> tags
    pattern = re.compile(r"<s>(.*?)</s>")
    matches = pattern.findall(input_string)
    return matches


def extract_array_result(input_string):
    # Use regular expression to find array result betweenn []
    # Assume input only have 1 array, or just want to get first array
    pattern = re.compile(
        r"\[(.*?)\]", re.DOTALL
    )  # Use DOTALL to handle multiline content
    matches = pattern.findall(input_string)
    if len(matches) > 0:
        return matches[0]
    return "[]"


def extract_string_array(input_string):
    text = input_string.strip()

    if not text or text == "[]":
        return []

    # Support callers that provide either the full bracketed form
    # or only the contents extracted from inside the brackets.
    text = text.strip("[]").strip()

    items = []

    for item in text.split(";"):
        cleaned = item.replace("\n", " ").strip()

        if not cleaned:
            continue

        # Remove matching wrapper quotes around an individual claim
        # while preserving quotation marks that are part of the claim.
        quote_pairs = [
            ('"', '"'),
            ("“", "”"),
            ("‘", "’"),
        ]

        for opening_quote, closing_quote in quote_pairs:
            if (
                cleaned.startswith(opening_quote)
                and cleaned.endswith(closing_quote)
                and len(cleaned) >= 2
            ):
                cleaned = cleaned[1:-1].strip()
                break

        items.append(cleaned)

    return items
