import subprocess
import sys
import os

cmd = [
    sys.executable, "-m", "evalplus.evaluate",
    "--model", "openai-community/gpt2",
    "--dataset", "humaneval",
    "--backend", "hf",
    "--greedy",
    "--defer-sanitize",
]

print("Python:", sys.executable)
print("CWD:", os.getcwd())
print("Running:", " ".join(cmd), flush=True)

result = subprocess.run(
    cmd,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)

print(result.stdout)
print("Return code:", result.returncode)

# Do not raise SystemExit yet while debugging.