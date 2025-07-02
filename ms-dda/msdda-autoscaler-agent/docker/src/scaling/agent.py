from kubernetes import client, config

DEPLOYMENT_NAME = "user"
NAMESPACE = "scaling"
config.load_incluster_config()

k8s_apps_v1 = client.AppsV1Api()

def scale_up(namespace_name=NAMESPACE, deployment_name=DEPLOYMENT_NAME):
    deployment = k8s_apps_v1.read_namespaced_deployment(
        name=deployment_name, namespace=namespace_name
    )
    if deployment is not None:
        if deployment.spec.replicas < 20:
            print(f"Current number of replicas: {deployment.spec.replicas}")
            deployment.spec.replicas = deployment.spec.replicas + 1

            print(f"Number of replicas after scale up: {deployment.spec.replicas}")

            current_cpu_requests = deployment.spec.template.spec.containers[
                0
            ].resources.requests["cpu"]
            current_cpu_limits = deployment.spec.template.spec.containers[
                0
            ].resources.limits["cpu"]            
            print(
                f"Current CPU request/limit: {current_cpu_requests}/{current_cpu_limits}"
            )

            current_cpu_requests = current_cpu_requests[0:current_cpu_requests.index("m")]
            current_cpu_limits = current_cpu_limits[0:current_cpu_limits.index("m")]

            deployment.spec.template.spec.containers[0].resources.requests[
                "cpu"
            ] = f"{int(float(current_cpu_requests) * 1.2)}m"

            deployment.spec.template.spec.containers[0].resources.limits[
                "cpu"
            ] = f"{int(float(current_cpu_limits) * 1.2)}m"

            api_response = k8s_apps_v1.patch_namespaced_deployment(
                name=deployment_name, namespace=namespace_name, body=deployment
            )

            print("Deployment updated. status='%s'" % str(api_response.status))
            return True
        else:
            print("Max number of replicas reached: 20")
            return False
    else:
        return False


def scale_down(namespace_name=NAMESPACE, deployment_name=DEPLOYMENT_NAME):
    deployment = k8s_apps_v1.read_namespaced_deployment(
        name=deployment_name, namespace=namespace_name
    )

    if deployment is not None:
        if deployment.spec.replicas > 1:
            print(f"Current number of replicas: {deployment.spec.replicas}")
            deployment.spec.replicas = deployment.spec.replicas - 1
            print(f"Number of replicas after scale down: {deployment.spec.replicas}")

            api_response = k8s_apps_v1.patch_namespaced_deployment(
                name=deployment_name, namespace=namespace_name, body=deployment
            )

            print("Deployment updated. status='%s'" % str(api_response.status))
            return True
        else:
            print("Min number of replicas reached: 1")
    else:
        return False


def get_replicas(namespace_name, deployment_name):
    deployment = k8s_apps_v1.read_namespaced_deployment(
        name=deployment_name, namespace=namespace_name
    )
    if deployment is not None:
        return deployment.spec.replicas
    else:
        return None


def get_deployment(namespace_name=NAMESPACE, deployment_name=DEPLOYMENT_NAME):
    deployment = k8s_apps_v1.read_namespaced_deployment(
        name=deployment_name, namespace=namespace_name
    )
    if deployment is not None:
        return deployment
    else:
        return None
