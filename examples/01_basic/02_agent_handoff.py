from gait import MAO, Agent, ObserverLoguru

# suffix = "gpt-4o-mini"  # os.environ["AZURE_API_DEPLOYMENT"]
# model = "azure/" + suffix
model = "ollama_chat/llama3.2:latest"
params = dict(
    # api_base=os.environ["AZURE_API_BASE"] + "/" + suffix,
    temperature=0.0,
    top_p=0.9,
)

agent_es = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish.",
    model=model,
    params=params,
)


def transfer_to_spanish_agent() -> Agent:
    """Transfer spanish speaking users immediately.
    """
    return agent_es


agent_en = Agent(
    name="English Agent",
    instructions="You only speak English.",
    functions=[transfer_to_spanish_agent],
    model=model,
    params=params,
)

if __name__ == "__main__":
    mao = MAO(agent_en, observer=ObserverLoguru())
    for _ in mao("Hola. ¿Como estás?"):
        if _.content:
            mao.terminate()
            print(_.content)
