# G.AI.T on FastAPI

This is a sample example of how to use G.AI.T on FastAPI.

```bash
uv pip install -U ".[fastapi]"
```

Create a `.env` file with the following sample content:

```dotenv
TITLE="G.AI.T"
LAYERS_JSON=~/data/WhereAssistant.json
AZURE_API_BASE=https://xxxx.azure-api.net/load-balancing/gpt-4o
AZURE_API_DEPLOYMENT=gpt-4o
AZURE_API_KEY=xxxx
AZURE_API_VERSION=2024-10-21
```

Execute the following command to start the FastAPI server:

```bash
fastapi run main.py --app app
```

or in background mode:

```bash
nohup fastapi run main.py --app app &> fastapi.log &
````

Open a browser and navigate to `http://localhost:8000/docs` to see the Swagger UI.

You can stop the process by executing:

```terminal
pkill -f fastapi
```
