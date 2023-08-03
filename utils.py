import re


def yield_lines(file_path, num_lines=10 ** 6):
    with open(file_path, 'r', encoding='latin-1') as file:
        lines = []
        for line in file:
            lines.append(line.strip())
            if len(lines) == num_lines:
                yield lines
                lines = []
        if lines:  # If there are remaining lines not yet yielded
            yield lines


def clean_text(text):
    text = text.strip()
    # Remove repeating double quotes
    text = re.sub(r'``+', r'"', text)
    # Replace curly single quotes and backticks with straight single quotes
    text = re.sub(r'[’‘`]', r"'", text)
    # Replace repeating single quotes with double quotes
    text = re.sub(r"''+", r'"', text)
    # Replace curly double quotes with straight double quotes
    text = re.sub(r'[“”]', r'"', text)
    # Remove repeating double quotes and replace with a single double quote
    text = re.sub(r'(")+', r'\1', text)
    # Add space before and after punctuation
    punctuation_pattern = "[-~!=\";:?+.,\(\)\\\/\*\[\]\}\{\|_^<>]"
    text = re.sub(punctuation_pattern, r" \g<0> ", text)
    return text
