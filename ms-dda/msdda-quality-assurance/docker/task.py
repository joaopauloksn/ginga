import threading
from datetime import datetime


def run_script():
    threading.Timer(5.0, run_script).start()
    print("Measuring autoscaler model quality...")

    return datetime.now()


print(f"Running Quality Assurance component script at {datetime.now()}")
run_script()
