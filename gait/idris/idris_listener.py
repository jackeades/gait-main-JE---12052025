from abc import ABC


class IdrisListener(ABC):
    """Abstract class to implement a listener interface.
    """

    def on_context(self, context: str) -> None:
        """Called when a context is added.

        :param context: The context.
        """
        pass

    def on_question_sql(self, question: str, sql: str) -> None:
        """Called when a question and SQL are added.

        :param question: The question.
        :param sql: The SQL.
        """
        pass


class IdrisListenerNoop(IdrisListener):
    """No-op implementation of the IdrisListener interface.
    """

    def on_context(self, context: str) -> None:
        return

    def on_question_sql(self, question: str, sql: str) -> None:
        return
