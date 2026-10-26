"""Numbered sources for one turn, shared across the agents that run in it.

Every piece of evidence an agent gathers (a textbook chunk, a web page, a SQL
result) gets a stable number. The answer cites those numbers as [n], and the
UI's citation panel maps them back to chunk IDs, URLs, or queries.
"""

MAX_SOURCE_CHARS = 6000  # default per source; whole sections pass a larger limit


class SourceRegistry:
    def __init__(self, existing: list[dict] | None = None):
        self.sources: list[dict] = [dict(s) for s in existing or []]

    def add(self, kind: str, key: str, title: str, text: str, url: str | None = None,
            max_chars: int = MAX_SOURCE_CHARS, context: str | None = None) -> int:
        """`text` is the source itself (shown in the UI); `context`, when given, is what the model
        reads instead, e.g. a digest of a whole section in a chapter summary."""
        for source in self.sources:
            if source["key"] == key:
                return source["n"]
        n = len(self.sources) + 1
        self.sources.append({
            "n": n,
            "kind": kind,  # "textbook", "web", or "sql"
            "key": key,  # chunk ID, section ID, URL, or query
            "title": title,
            "url": url,
            "text": text[:max_chars],
            **({"context": context} if context else {}),
        })
        return n

    def render(self, numbers: list[int] | None = None) -> str:
        chosen = [s for s in self.sources if numbers is None or s["n"] in numbers]
        return "\n\n".join(f"[{s['n']}] {s['title']}\n{s.get('context') or s['text']}" for s in chosen) or "(no sources)"
