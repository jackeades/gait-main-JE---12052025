from gait import Agent


if __name__ == "__main__":
    # Create a new agent.
    agent = Agent()
    
    # Ask the agent a question.
    resp = agent("Please explain in one VERY short sentence 'What is the number 42?'")
    
    # Print the response.
    print(resp.content)
