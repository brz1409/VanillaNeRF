import os
import hashlib
import subprocess

# Benutzername aus Umgebungsvariable holen
username = os.getenv('USER')

# Port berechnen (z. B. 63000 + Hash des Benutzernamens)
user_hash = int(hashlib.sha256(username.encode()).hexdigest(), 16) % 100
port = 63000 + (int(hashlib.sha256(username.encode()).hexdigest(), 16) % 1000)


# TensorBoard-Befehl vorbereiten
logdir = "./runs"
cmd = [
    "tensorboard",
    f"--logdir={logdir}",
    f"--port={port}",
    "--bind_all"
]

# Optional: URL ausgeben (z. B. für JupyterHub auf VSC)
url = f"https://jupyter.datalab.tuwien.ac.at/user/{username}/proxy/{port}/"
print(f"TensorBoard läuft unter: {url}")

# TensorBoard starten
subprocess.Popen(cmd)

print("TensorBoard-Prozess wurde gestartet.")