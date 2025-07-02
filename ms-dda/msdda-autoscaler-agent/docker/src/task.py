import threading
from datetime import datetime
import model_deployment.deploy as deploy


def run_script():
    threading.Timer(20.0, run_script).start()
    print("############### Scaling task execution information ###############")

    deploy.deploy_model()


print(f"Running Autoscaler agent component script at {datetime.now()}")
run_script()
