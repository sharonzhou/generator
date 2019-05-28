from sys import stderr
import argparse

def print_err(*args, **kwargs):
    """Print a message to stderr."""
    print(*args, file=stderr, **kwargs)

def args_to_list(csv, allow_empty, arg_type=int, allow_negative=True):
    """Convert comma-separated arguments to a list. Only take non-negative values."""
    arg_vals = [arg_type(d) for d in str(csv).split(',')]
    if not allow_negative:
        arg_vals = [v for v in arg_vals if v >= 0]
    if not allow_empty and len(arg_vals) == 0:
        return None
    return arg_vals

def str_to_bool(v):
    """Convert an argument string into its boolean value."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
