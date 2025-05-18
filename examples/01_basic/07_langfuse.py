#
# https://langfuse.com/docs/deployment/local
# pip install langfuse
#
import litellm

from gait import MAO, Agent

litellm.success_callback = ["langfuse"]

if __name__ == "__main__":
    mao = MAO(Agent())
    print(mao("Please explain in one short sentence what is telemetry and call tracing?"))
