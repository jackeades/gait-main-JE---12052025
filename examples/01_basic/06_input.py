from gait import MAO, Agent

if __name__ == "__main__":
    mao = MAO(Agent(
        # model=f"azure/{os.environ['AZURE_API_DEPLOYMENT']}",
    ))
    while (user_input := input("Enter something (type 'stop' to exit): ")) != "stop":
        for _ in mao(user_input):
            if _.content:
                mao.terminate()
                print(_.content)
