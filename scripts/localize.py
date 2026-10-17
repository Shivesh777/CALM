"""Rewrite the cloud working directory used in the job files (/kaggle/working) to a local directory.

  python scripts/localize.py /path/to/workdir     # edits experiments/pre/*.sh and experiments/jobs/*.jsonl in place
"""
import glob
import os
import sys

work = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
n = 0
for f in glob.glob(os.path.join(root, "pre", "*.sh")) + glob.glob(os.path.join(root, "jobs", "*.jsonl")):
    s = open(f).read()
    if "/kaggle/working" in s:
        open(f, "w").write(s.replace("/kaggle/working", work))
        n += 1
print(f"rewrote {n} files: /kaggle/working -> {work}")
