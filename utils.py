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
    import numpy as np
    binary_data = np.memmap(path, dtype=np.uint16, mode='r')
    print('Total tokens: ', len(binary_data) // 1024 ** 2, 'M')
    num_batches = len(binary_data) // seq_len
    return num_batches
