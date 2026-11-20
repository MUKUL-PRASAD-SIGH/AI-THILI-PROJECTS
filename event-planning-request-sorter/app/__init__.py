"""Event Planning Request Sorter."""
import os

# This project only uses PyTorch. If TensorFlow happens to be installed,
# stop transformers from importing it (slow, noisy, and can crash on Windows).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
