"""Where the package keeps its data, the shipped tokenizers and its cache."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
PRETRAINED_ROOT = os.path.join(HERE, "pretrained")
CACHE_DIR = os.environ.get("PAPAYA_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "papaya"))


def available():
    """names of the tokenizers shipped with the package"""
    return sorted(d for d in os.listdir(PRETRAINED_ROOT)
                  if os.path.exists(os.path.join(PRETRAINED_ROOT, d, "tokenizer.json")))


def pretrained_dir(name_or_path):
    """a shipped tokenizer name (papaya-v2-16k) or a path to a tokenizer dir"""
    if os.path.isdir(name_or_path):
        return name_or_path
    d = os.path.join(PRETRAINED_ROOT, name_or_path)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{name_or_path!r} is neither a directory nor a shipped tokenizer "
                                f"({', '.join(available()) or 'none installed'})")
    return d
