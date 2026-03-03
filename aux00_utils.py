"""
Utility functions for APE analysis
"""

import time
from functools import wraps


def timeit(func):
    """Decorator that prints the elapsed time of a function call"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n{func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Elapsed wall time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        return result
    return wrapper
