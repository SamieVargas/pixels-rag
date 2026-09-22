"""Follow-ups. A session keeps the last three turns and hands them to the
router, which rewrites "and on weekends?" into a standalone question before
anything is retrieved (the rewritten_query field of the router contract).
State lives in the session object and nowhere else."""

from core.pipeline import ask

KEEP_TURNS = 3


class Session:
    def __init__(self, *, days, collection, client, keep: int = KEEP_TURNS, **ask_kwargs):
        self.days, self.collection, self.client, self.keep, self.ask_kwargs = days, collection, client, keep, ask_kwargs
        self.turns: list[dict] = []

    def history(self) -> list[dict]:
        """The last `keep` question-and-answer pairs, oldest first."""
        return self.turns[-2 * self.keep:]

    def ask(self, question: str, *, rewrite: bool = True) -> dict:
        history = self.history() if rewrite else None
        r = ask(question, days=self.days, collection=self.collection, client=self.client, history=history or None, **self.ask_kwargs)
        self.turns.append({"role": "user", "text": question})
        self.turns.append({"role": "assistant", "text": r["answer"]["answer"]})
        return r
