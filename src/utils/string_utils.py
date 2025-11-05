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
    items = input_string.strip("[").strip("]").strip('"').strip().split(";")
    # remove empty string in list and trim \n
    items = [i.replace("\n", "") for i in items if i]
    return items
