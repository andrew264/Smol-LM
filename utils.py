import os


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


def enable_memory_growth():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # enable cuda_malloc_async allocator
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    print(f"Enabled TF_GPU_ALLOCATOR: {os.environ['TF_GPU_ALLOCATOR']}")
