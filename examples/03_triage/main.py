import os

from gait import Agent, MAO, ObserverLoguru

model = "azure/" + os.environ["OPENAI_API_DEPLOYMENT"]
params = {
    "temperature": 0.0,
    "top_p": 0.9,
}


def transfer_to_triage() -> Agent:
    """Transfer to the triage agent.
    """
    return triage_agent


sales_agent = Agent(
    name="Sales Agent",
    model=model,
    params=params,
    instructions="You are a sales agent. You can help with sales inquiries.",
    functions=[transfer_to_triage],
)


def transfer_to_sales() -> Agent:
    """Transfer to the sales agent.
    """
    return sales_agent


def issue_refund(order_number: str) -> str:
    """Issue a refund for an order.

    :param order_number: The order number a refund for.
    """
    return f"Refund issued for {order_number}. Please allow 3-5 business days for the refund to appear in your account."


refunds_agent = Agent(
    name="Refunds Agent",
    model=model,
    params=params,
    instructions="You are a refunds agent. You can help with refunds.",
    functions=[issue_refund, transfer_to_triage],
)


def transfer_to_refunds() -> Agent:
    """Transfer to the refund's agent.
    """
    return refunds_agent


triage_agent = Agent(
    name="Triage Agent",
    model=model,
    params=params,
    instructions="You are a triage agent. You can help with triaging inquiries.",
    functions=[transfer_to_sales, transfer_to_refunds],
)

if __name__ == "__main__":
    mao = MAO(triage_agent)
    mao(f"I would like a refund for order 12-34.")
    # pprint(assistant.messages, expand_all=True)
