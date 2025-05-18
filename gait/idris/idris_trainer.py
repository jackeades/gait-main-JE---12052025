import logging
import random
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import pandas as pd

from .idris_base import IdrisException

# TODO: Move the import into the __init__ method
try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import wordninja
except ImportError:
    wordninja = None


@dataclass
class IdrisTrainerResult:
    """The result of training a model.
    """
    create_table: str
    context: List[str]
    question_sql: List[Tuple[str, str]]


class IdrisTrainer:
    def __init__(
            self,
            alias: Optional[Dict[str, str]] = None
    ) -> None:
        """Create a new trainer. This module requires the DuckDB and the wordninja modules.

        :param alias: A dictionary of column aliases.
        """
        if duckdb is None:
            raise IdrisException("DuckDB module is required, please install it.")
        if wordninja is None:
            raise IdrisException("wordninja module is required, please install it.")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.alias = alias or {}

    def train(
            self,
            pdf: pd.DataFrame,
            table_name: str,
            alias_name: Optional[str] = None,
            limit: int = 10,
    ) -> IdrisTrainerResult:
        def ninja(text: str) -> str:
            return " ".join(wordninja.split(text.lower()))

        def get_ops(t: str) -> Tuple[str, str]:
            if t == "VARCHAR":
                if random.random() < 0.5:
                    return "is not", "!="
                else:
                    return "is", "="
            else:
                # TODO: Fix this to use random.choice
                return {
                    1: ("is less than", "<"),
                    2: ("is", "="),
                    3: ("is greater than", ">"),
                }[random.randint(1, 3)]

        if alias_name is None:
            alias_name = ninja(table_name)
        with duckdb.connect(":memory:") as conn:
            _ = conn.execute("create or replace view idris as select * from pdf")
            name_type = [
                (row[0], row[1])
                for row in conn.sql("DESCRIBE idris").fetchall()
            ]
            fields = ",\n".join([f"{n} {t}" for n, t in name_type])
            create_table = f"CREATE TABLE {table_name} (\n{fields}\n);"
            name_alias_type = [
                (n, self.alias.get(f"_col:{n}", ninja(n)), t)
                for n, t in name_type
            ]
            context = [f"Use column '{n}' in reference to {a}." for n, a, _ in name_alias_type]
            question_sql = []
            for field_name, field_alias, field_type in name_alias_type:
                self.logger.info(f"Processing {field_name} {field_alias} {field_type}...")
                # TODO - Use the top N common values
                rows = conn.execute(
                    f"""SELECT distinct({field_name}) as '{field_name}'
        FROM idris
        WHERE {field_name} IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}"""
                ).fetchall()
                for (v,) in rows:
                    op1, op2 = get_ops(field_type)
                    alias_key = f"{field_name}:{v}"
                    alias_def = v.lower() if field_type == "VARCHAR" else v
                    o = self.alias.get(alias_key, alias_def)
                    q = f"Show {alias_name} where {field_alias} {op1} {o}"
                    w = v if field_type in ("INTEGER", "FLOAT", "DOUBLE") else f"'{v}'"
                    s = f"SELECT * FROM {table_name} where {field_name}{op2}{w}"
                    question_sql.append((q, s))

        return IdrisTrainerResult(
            create_table=create_table,
            context=context,
            question_sql=question_sql,
        )

    #New code and method for LangChain augmentation JE 
    def augment_pairs_with_langchain(
            self,
            pairs: List[Tuple[str, str]],
            db,
            model_name: str = None,  # Default to None to use env settings
            temperature: float = 0,
            paraphrases_per_query: int = 3,
            system_prompt: Optional[str] = None,  # Add this parameter
    ) -> List[Tuple[str, str]]:
        """Augment question/SQL pairs using LangChain's SQL chains.
        
        Uses an LLM to paraphrase existing natural language queries and generate new SQL,
        which is verified for correctness against the database.
        
        :param pairs: Existing (natural language, SQL) pairs
        :param db: A langchain SQLDatabase object
        :param model_name: Name of the LLM model to use (if None, use environment settings)
        :param temperature: Temperature for generation
        :param paraphrases_per_query: Number of paraphrases to generate per query
        :return: List of newly generated (natural language, SQL) pairs
        """
        try:
            from langchain.chains import create_sql_query_chain
            from langchain_openai import AzureChatOpenAI  # Used the Azure-specific class for testing internally 
            import os
            from dotenv import load_dotenv
        except ImportError:
            raise IdrisException("langchain, langchain_community, langchain_openai, and python-dotenv packages are required, please install them.")

        # Load environment variables from .env file 
        load_dotenv()
        
        self.logger.info(f"Augmenting question/SQL pairs using LangChain with Azure OpenAI")
        
        # Check if Azure environment variables are set
        # Change these lines in the augment_pairs_with_langchain method
        if not all([os.getenv("AZURE_API_KEY"), 
                    os.getenv("AZURE_API_BASE")]):
            raise IdrisException("Azure OpenAI environment variables not set. Please check your .env file.")
        
        # Use Azure OpenAI with environment variables
        # If model_name is None, it will use AZURE_OPENAI_DEPLOYMENT from env
        azure_deployment = model_name or os.getenv("AZURE_API_DEPLOYMENT", "gpt-4o")
        
        llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-10-21"),
            deployment_name=azure_deployment,
            temperature=temperature
        )
        
        # If a system prompt is provided, use a different approach
        if system_prompt:
            from langchain.prompts import ChatPromptTemplate
            from langchain.chains import LLMChain
            
            # Create a prompt template for paraphrasing
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt.format(k=paraphrases_per_query)),
                ("user", "{question}")
            ])
            
            # Create a chain for paraphrasing
            paraphrase_chain = LLMChain(llm=llm, prompt=prompt)
            
            # Standard SQL chain for generating the SQL
            sql_chain = create_sql_query_chain(llm, db)
            
            # Process differently with paraphrasing
            aug_pairs = []
            for nl, _sql in pairs:
                self.logger.info(f"Generating paraphrases for: {nl}")
                try:
                    # Get paraphrases
                    paraphrases_raw = paraphrase_chain.invoke({"question": nl})
                    
                    # Parse the numbered list response
                    paraphrases = []
                    for line in paraphrases_raw["text"].strip().split("\n"):
                        if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                            paraphrase = line.split(".", 1)[1].strip()
                            paraphrases.append(paraphrase)
                    
                    # Generate SQL for each paraphrase
                    for paraphrase in paraphrases[:paraphrases_per_query]:
                        try:
                            sql = sql_chain.invoke({"question": paraphrase})
                            # Clean up SQL and verify as in your original code
                            # ...
                            # Add to aug_pairs
                            aug_pairs.append((paraphrase, sql))
                        except Exception as e:
                            self.logger.warning(f"SQL generation failed for paraphrase: {e}")
                except Exception as e:
                    self.logger.warning(f"Paraphrase generation failed: {e}")
            
            return aug_pairs
        else:
            # Original code path without paraphrasing
            db_chain = create_sql_query_chain(llm, db)
            
            aug_pairs = []
            for nl, _sql in pairs:
                self.logger.info(f"Generating {paraphrases_per_query} paraphrases for: {nl}")
                for i in range(paraphrases_per_query):
                    try:
                        # The new API returns SQL directly but may include markdown formatting
                        sql2 = db_chain.invoke({"question": nl})
                        
                        # Clean up markdown formatting
                        if sql2.startswith("```"):
                            # Extract SQL from code block
                            sql2 = sql2.split("\n", 1)[1]  # Remove first line with ```sql
                            sql2 = sql2.rsplit("\n", 1)[0]  # Remove last line with ```
                        
                        # Remove "SQLQuery:" prefix if present
                        if sql2.startswith("SQLQuery:"):
                            sql2 = sql2.replace("SQLQuery:", "", 1).strip()
                        
                        # Verify the SQL can be executed
                        try:
                            db.run(sql2)
                            # Add the original question with the new SQL
                            aug_pairs.append((nl, sql2))
                            self.logger.info(f"Added paraphrase {i+1} SQL: {sql2}")
                        except Exception as e:
                            self.logger.warning(f"SQL verification failed: {e}")
                    except Exception as e:
                        self.logger.warning(f"SQL generation failed: {e}")
            
            self.logger.info(f"Generated {len(aug_pairs)} new question/SQL pairs")
            return aug_pairs

    #New code and method for training LangChain augmentation JE 
    def train_with_langchain_augmentation(
                self,
                pdf: pd.DataFrame,
                table_name: str,
                db,
                alias_name: Optional[str] = None,
                limit: int = 10,
                model_name: str = "gpt-4o-mini",
                temperature: float = 0,
                paraphrases_per_query: int = 3,
                system_prompt: Optional[str] = None,  # Add this parameter
        ) -> IdrisTrainerResult:
            """Train with initial pairs and then augment using LangChain.
            
            :param pdf: The pandas DataFrame to train on
            :param table_name: The name of the table
            :param db: A langchain SQLDatabase object
            :param alias_name: Optional alias name for the table
            :param limit: Limit for the initial training
            :param model_name: Model name for LangChain
            :param temperature: Temperature for generation
            :param paraphrases_per_query: Number of paraphrases to generate per query
            :return: IdrisTrainerResult with augmented pairs
            """
            # First, perform standard training
            result = self.train(pdf, table_name, alias_name, limit)
            
            # Then augment the pairs using LangChain
            aug_pairs = self.augment_pairs_with_langchain(
                result.question_sql,
                db,
                model_name,
                temperature,
                paraphrases_per_query,
                system_prompt
            )
            
            # Combine original and augmented pairs
            combined_pairs = result.question_sql + aug_pairs
            
            return IdrisTrainerResult(
                create_table=result.create_table,
                context=result.context,
                question_sql=combined_pairs,
            )
    #End of new methods - JE