import json
import logging
import os.path
import re
import warnings
from typing import Optional, List, Tuple

import pandas as pd
import sqlparse

from .idris_abc import IdrisABC
from .idris_emb import IdrisEmb
from .idris_listener import IdrisListener, IdrisListenerNoop
from .idris_llm import IdrisLLM
from .idris_rdb import IdrisRDB


class IdrisException(Exception):
    """Base class for exceptions in Idris.
    """
    pass


DEFAULT_IDRIS_SYSTEM_1 = """You are an AI expert and helpful assistant specialising in {dialect} SQL.
Answer the questions by providing SQL code that is compatible with a {dialect} environment.
This is the question you are required to answer:
{prompt}

===DDL statement(s):
{create_table}

===Additional Context:
{context}
{topics}

===Please follow the below response guidelines:
- If the provided context is sufficient, please generate a valid SQL query without any explanations for the question.
- If the provided context is insufficient, please explain why it can't be generated.
- Please use the most relevant table(s).
- If the question has been asked and answered before, please REPEAT the answer EXACTLY as it was given before.
- Please ensure that the output SQL is {dialect} compliant, executable and free of syntax errors.
- Make sure to NOT add any markdown formatting or tags.
- If you generate the correct SQL query, you will receive a reward of $1,000,000.
"""

DEFAULT_IDRIS_SYSTEM_2 = """You are an AI expert and helpful assistant specializing in {dialect} SQL.  
Your task is to answer the following question by providing SQL code compatible with a {dialect} environment:  
{prompt}

===DDL statement(s):  
{create_table}

===Additional Context:  
{context}  
{topics}

===Response Guidelines:
- If the provided DDL and context are sufficient to answer the question, generate a valid {dialect} SQL query without explanations.
- If the DDL or context is insufficient to generate a query, explain clearly why the SQL cannot be produced (e.g., missing table definitions, ambiguous requirements).
- Use the most relevant table(s) from the provided DDL based on the question.
- If this exact question has been asked and answered previously, repeat the prior SQL answer verbatim.
- Ensure the generated SQL is fully compliant with {dialect}, executable, and free of syntax errors.
- Do not include markdown formatting, code tags, or extraneous text in the response.
- A correct and executable SQL query will earn a reward of $1,000,000.
- If the user's wording is ambiguous, *infer common synonyms or paraphrases* \
  before rejecting the request.
"""


class Idris(IdrisABC):
    def __init__(
            self,
            idris_rdb: IdrisRDB,
            idris_emb: IdrisEmb,
            idris_llm: IdrisLLM,
            system: Optional[str] = None,
            topics: Optional[str] = None,
            listener: Optional[IdrisListener] = None,
    ) -> None:
        """Initialise an Idris instance.

        :param idris_rdb: A IdrisRDB instance.
        :param idris_emb: A IdrisEmb instance.
        :param idris_llm: A IdrisLLM instance.
        :param system: An optional system message to override default one.
        :param topics: An optional topics message to add to the context.
        :param listener: An optional listener to listen for context and question/sql.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.idris_rdb = idris_rdb
        self.idris_emb = idris_emb
        self.idris_llm = idris_llm
        # https://medium.com/datamindedbe/prompt-engineering-for-a-better-sql-code-generation-with-llms-263562c0c35d
        self.system = system or DEFAULT_IDRIS_SYSTEM_2
        self.topics = topics or ""
        self.listener = listener or IdrisListenerNoop()

    def close(self) -> None:
        """Close open resources.
        """
        self.idris_rdb.close()
        self.idris_emb.close()
        self.idris_llm.close()

    def add_context(
            self,
            context: str
    ) -> None:
        """Add context to an Idris instance.

        :param context: The context to add.
        """
        self.idris_emb.add_context(context)

    def load_context(
            self,
            context: List[str]
    ) -> None:
        """Load context from a list.

        :param context: A list of context.
        """
        self.idris_emb.load_context(context)

    def dump_context(self) -> List[str]:
        """Dump the context.

        :return: The context.
        """
        return self.idris_emb.dump_context()

    def load_context_json(
            self,
            filename: Optional[str] = None,
    ) -> None:
        """Load context from a JSON file.

        :param filename: The filename of the JSON file. If None use "~/context.json".
        """
        if filename is None:
            filename = os.path.join("~", "context.json")
        filename = os.path.expanduser(filename)
        if not os.path.exists(filename):
            raise IdrisException(f"File not found: {filename}")
        with open(filename, mode="r", encoding="utf-8") as fp:
            self.load_context(json.load(fp))

    def add_question_sql(
            self,
            question: str,
            sql: str
    ) -> None:
        """Add a question and its corresponding SQL to an Idris instance.

        :param question: The question.
        :param sql: The SQL.
        """
        self.idris_emb.add_question_sql(question, sql)

    def load_question_sql(
            self,
            question_sql: List[Tuple[str, str]],
    ) -> None:
        """Load question and SQL from a list.

        :param question_sql: A list of question and SQL.
        """
        self.idris_emb.load_question_sql(question_sql)

    def dump_question_sql(self) -> List[Tuple[str, str]]:
        """Dump the question and SQL.

        :return: The question and SQL.
        """
        return self.idris_emb.dump_question_sql()

    def load_question_sql_json(
            self,
            filename: Optional[str] = None,
    ) -> None:
        """Load Question/SQL from a JSON file.

        :param filename: The filename of the JSON file. If None use "~/question_sql.json".
        """
        if filename is None:
            filename = os.path.join("~", "question_sql.json")
        filename = os.path.expanduser(filename)
        if not os.path.exists(filename):
            raise IdrisException(f"File not found: {filename}")
        with open(filename, mode="r", encoding="utf-8") as fp:
            self.load_question_sql(json.load(fp))

    def add_create_table(
            self,
            create_table: str
    ) -> None:
        """Add a "CREATE TABLE" statement to an Idris instance.

        :param create_table: The "CREATE TABLE" statement.
        """
        self.idris_emb.add_create_table(create_table)

    def dump_create_table(self) -> List[str]:
        """Dump the "CREATE TABLE" statements.

        :return: The "CREATE TABLE" statements.
        """
        return self.idris_emb.dump_create_table()

    def add_create_table_sql(
            self,
            filename: Optional[str] = None,
    ) -> None:
        """Add CREATE TABLE statement from a SQL text file.

        :param filename: The filename of the SQL text file. If None use "~/create_table.sql".
        """
        warnings.warn("Method 'add_create_table_sql' is deprecated. Use 'load_create_table_sql' instead.",
                      DeprecationWarning)
        self.load_create_table_sql(filename)

    def load_create_table_sql(
            self,
            filename: Optional[str] = None,
    ) -> None:
        """Add CREATE TABLE statement from a SQL text file.

        :param filename: The filename of the SQL text file. If None use "~/create_table.sql".
        """
        if filename is None:
            filename = os.path.join("~", "create_table.sql")
        filename = os.path.expanduser(filename)
        if not os.path.exists(filename):
            raise IdrisException(f"File not found: {filename}")
        # Read the file and split by ';'
        with open(filename, mode="r", encoding="utf-8") as fp:
            text = fp.read()
            for split in text.split(";"):
                self.add_create_table(split.strip())

    def add_describe_table(
            self,
            table_name: str,
    ) -> None:
        """Add a table name and let IDRIS describe the table and get CREATE TABLE statement.

        :param table_name: The name of the table.
        """
        create_table = self.idris_rdb.get_create_table(table_name)
        if create_table is not None:
            self.idris_emb.add_create_table(create_table)

    def generate_sql(
            self,
            prompt: str,
            **kwargs,
    ) -> Optional[str]:
        """Generate SQL for a given a user prompt.

        :param prompt: The user prompt.
        :return: The generated SQL or None if no SQL could be generated.
        """
        self.logger.info(f"User Question: {prompt}")
        messages = []

        create_table = "\n".join([_ for _ in self.idris_emb.get_similar_create_table(prompt)])

        self.logger.info("Looking for similar context...")
        context = []
        for _ in self.idris_emb.get_similar_context(prompt):
            self.listener.on_context(_)
            self.logger.info(_)
            context.append(_)
        context = "\n".join(context)

        system = self.system.format(
            prompt=prompt,
            dialect=self.idris_rdb.dialect,
            create_table=create_table,
            context=context,
            topics=self.topics,
        )
        messages.append(
            self.idris_llm.create_message(role="system", content=system)
        )

        self.logger.info("Looking for similar question/sql...")
        for question, sql in self.idris_emb.get_similar_question_sql(prompt):
            self.listener.on_question_sql(question, sql)
            self.logger.info(f"{question} --> {sql}")
            messages.append(
                self.idris_llm.create_message(
                    role="user",
                    content=question)
            )
            messages.append(
                self.idris_llm.create_message(
                    role="assistant",
                    content=re.sub(r"\s*\n\s*", " ", sql).strip())
            )

        # Add the user question.
        messages.append(
            self.idris_llm.create_message(role="user", content=prompt)
        )

        sql = self.idris_llm.generate_sql(messages, **kwargs)
        if sql is not None:
            sql = re.sub(r"\s*\n\s*", " ", sql).strip()
        self.logger.info(f"Final Answer: {sql}")
        return sql

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL.

        :param sql: The SQL to execute.
        :return: The result of the SQL as a Pandas DataFrame.
        """
        return self.idris_rdb.execute_sql(sql)

    def __call__(self, prompt: str, **kwargs) -> Optional[pd.DataFrame]:
        """Generate SQL for a given a user prompt and execute it.

        :param prompt: The user prompt.
        :return: A pandas dataframe with the result of the SQL if valid.
        """
        sql = self.generate_sql(prompt, **kwargs)
        sql = sql.replace("```sql", "").replace("```", "").strip()
        if is_sql_valid(sql):
            return self.execute_sql(sql)
        else:
            self.logger.warning(f"Invalid SQL {sql}.")
            return None

    def load_from_path(self, path: str) -> None:
        """
        Load the context, create_table and question_sql from a path.

        :param path: The folder path containing the context, create_table and question_sql files.
        """
        self.load_create_table_sql(os.path.join(path, "create_table.sql"))
        self.load_context_json(os.path.join(path, "context.json"))
        self.load_question_sql_json(os.path.join(path, "question_sql.json"))

    def augment_question_sql_with_langchain(
            self,
            db,
            model_name: str = "gpt-4o-mini",
            temperature: float = 0,
            paraphrases_per_query: int = 3,
    ) -> None:
        """Augment existing question/SQL pairs using LangChain.
        
        :param db: A langchain SQLDatabase object
        :param model_name: Name of the LLM model to use
        :param temperature: Temperature for generation
        :param paraphrases_per_query: Number of paraphrases per query
        """
        from gait.idris.idris_trainer import IdrisTrainer
        
        trainer = IdrisTrainer()
        existing_pairs = self.dump_question_sql()
        
        aug_pairs = trainer.augment_pairs_with_langchain(
            existing_pairs,
            db,
            model_name,
            temperature,
            paraphrases_per_query
        )
        
        # Load the augmented pairs
        self.load_question_sql(aug_pairs)


def is_sql_valid(sql: str) -> bool:
    """Check if the SQL is valid; Starts with a SELECT keyword.

    :param sql: The SQL to check.
    :return: True if the SQL is valid, False otherwise.
    """
    is_valid = False
    if sql is not None:
        statements = sqlparse.parse(sql)
        for statement in statements:
            if statement.get_type() == "SELECT":
                is_valid = True
                break
            break

    return is_valid


def get_query_definition(
        sql: str,
        default: Optional[str] = None,
) -> Optional[str]:
    """Get the query definition from the SQL; The WHERE value if any.

    :param sql: The SQL to parse.
    :param default: The default value to return if not found.
    :return: The query definition, or None if not found.
    """
    query_def = default
    if sql is not None:
        sql = sql.replace("```", "")
        sql = re.sub(r'\s*\n\s*', " ", sql).strip()
        statements = sqlparse.parse(sql)
        for statement in statements:
            for token in statement:
                s = str(token)
                if s[:6].upper() == "WHERE ":
                    query_def = re.sub(r'\s*;$', "", s[6:])
                    break
            break

    return query_def
