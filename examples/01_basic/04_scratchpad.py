#
# This example demonstrates how to use the scratchpad to store and retrieve information during the conversation.
# The scratchpad is a dictionary that can be used to store information that can be used by the agent or the functions.
#
from gait import Agent, MAO, Scratchpad


def instructions(scratchpad: Scratchpad) -> str:
    """Create a dynamic system instructions for the agent from information in the scratchpad.

    :param scratchpad: A scratchpad instance.
    """

    # Get the name from the scratchpad or use a default value.
    name = scratchpad.get("name", "User")
    return f"You are a helpful agent. Greet the user by his name ({name})."


def report_account_details(scratchpad: Scratchpad) -> str:
    """Report the account details for a user.

    :param scratchpad: A scratchpad instance. Implicitly passed by the MAO.
    """
    name = scratchpad.get("name", "Unknown")
    balance = scratchpad.get("balance", 12.45)
    return f"Account Details: {name=} {balance=}"


if __name__ == "__main__":

    # Create a new agent with a model and a set of functions.
    agent = Agent(
        model="azure/gpt-4o-mini",
        instructions=instructions,
        functions=[report_account_details],
    )

    mao = MAO(agent)
    # Inject into the scratchpad the name of the user.
    mao.scratchpad["name"] = "Mansour"

    # Start the conversation.
    for _ in mao("Hi there!"):
        mao.terminate()
        print(_.content)

    for _ in mao("Please report the balance on my account."):
        if _.content:
            mao.terminate()
            print(_.content)
