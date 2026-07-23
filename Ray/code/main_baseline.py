from ray import serve

from llm_deployment import BaseLLMDeployment
import config


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": config.static_max_replicas(),
        "target_num_ongoing_requests_per_replica": 2,
    },
    ray_actor_options={"num_cpus": 1},
)
class BaselineLLMDeployment(BaseLLMDeployment):
    def __init__(self, controller_handle=None):
        super().__init__(controller_handle=None)


app = BaselineLLMDeployment.bind()


if __name__ == "__main__":
    import ray
    import signal
    import sys

    ray.init(ignore_reinit_error=True)

    serve.run(app, route_prefix="/chat")

    print("Baseline Ray Serve app is running at http://localhost:8000/chat")

    try:
        input("Press Enter to stop...\n")
    except (EOFError, KeyboardInterrupt):
        if sys.stdin.isatty():
            print("Shutting down...")
            sys.exit(0)

        try:
            signal.pause()
        except (KeyboardInterrupt, AttributeError):
            pass

    print("Shutting down...")
