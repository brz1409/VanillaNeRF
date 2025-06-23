"""Convenience script to launch TensorBoard with a predictable port."""

import os
import hashlib
import subprocess

# Determine the current user so the port can be derived from the username.
username = os.getenv("USER")

# Port berechnen (z. B. 63000 + Hash des Benutzernamens)
user_hash = int(hashlib.sha256(username.encode()).hexdigest(), 16) % 100
port = 63000 + (int(hashlib.sha256(username.encode()).hexdigest(), 16) % 1000)


# Assemble the TensorBoard command. ``--bind_all`` makes the interface
# accessible when running on a remote machine such as a JupyterHub server.
logdir = "./runs"
cmd = [
    "tensorboard",
    f"--logdir={logdir}",
    f"--port={port}",
    "--bind_all"
]

# Optional: URL ausgeben (z. B. für JupyterHub auf VSC)
# Construct a convenient URL that can be opened in the browser when running on
# TU Wien's JupyterHub infrastructure. Adjust for your own setup if needed.
url = f"https://jupyter.datalab.tuwien.ac.at/user/{username}/proxy/{port}/"
print(f"TensorBoard läuft unter: {url}")

# Launch TensorBoard in a background process so the script exits immediately.
subprocess.Popen(cmd)

print("TensorBoard-Prozess wurde gestartet.")