import glob
import os
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


def multiline_input(prompt: str = '>> ') -> str:
    lines = []
    while True:
        line = input(prompt).strip()
        if line:
            lines.append(line)
        else:
            break
    return '\n'.join(lines)


def get_total_steps(path: str, seq_len: int) -> int:
    import tensorflow as tf
    files = glob.glob(path, recursive=True)
    steps = 0
    for file_path in files:
        binary_data = tf.io.read_file(file_path)
        m = tf.io.decode_raw(binary_data, tf.uint16)
        num_batches: tf.Tensor = tf.shape(m)[0] // (seq_len + 1)
        steps += num_batches.numpy()
    return steps


def enable_memory_growth():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # enable cuda_malloc_async allocator
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    print(f"Enabled TF_GPU_ALLOCATOR: {os.environ['TF_GPU_ALLOCATOR']}")


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
