from gait import Idris, IdrisNoRDB, IdrisLiteLLM
from gait.idris_litellm import IdrisLiteEmb

if __name__ == "__main__":
    with IdrisNoRDB() as rdb:
        with IdrisLiteEmb(
                model_name="ollama/mxbai-embed-large:latest",
        ) as emb:
            with IdrisLiteLLM(
                    model_name="ollama/phi4:14b-q8_0",
            ) as llm:
                with Idris(rdb, emb, llm) as idris:
                    pass
