"""The regulation as a read-only directory tree over the chunk table.

Chunk paths (MDR/Kapitel-II/Art-10/Abs-9) already encode the hierarchy, so listing
is a prefix query and reading reassembles the chunks under a path in order.
"""

import psycopg

from mdr_rag.retrieval import Lang

READ_LIMIT = 4000


def _under(path: str) -> str:
    return path.rstrip("/").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"


def list_children(conn: psycopg.Connection, lang: Lang, path: str) -> list[str]:
    """One line per child: name, '/' if it has children, title for articles and annexes."""
    path = path.rstrip("/")
    rows = conn.execute(
        "SELECT path, heading FROM chunks WHERE lang = %s AND path LIKE %s ORDER BY position",
        (lang, _under(path)),
    ).fetchall()
    children: dict[str, str] = {}
    for row in rows:
        rest = row["path"][len(path) + 1 :].split("/")
        name = rest[0] + ("/" if len(rest) > 1 else "")
        unit_level = name.startswith(("Art-", "Anhang-", "Annex-"))
        title = " – ".join(row["heading"].split(" – ")[:2])  # without annex chapter titles
        children.setdefault(name, f"  {title}" if unit_level else "")
    # A path can be both a chunk and a directory (an article with its own intro text).
    names = [n for n in children if n.rstrip("/") + "/" not in children or n.endswith("/")]
    return [f"{name}{children[name]}" for name in names]


def read(conn: psycopg.Connection, lang: Lang, path: str) -> tuple[str, list[str]]:
    """Text of a path and everything below it, plus the chunk paths it contains."""
    path = path.rstrip("/")
    rows = conn.execute(
        """
        SELECT path, heading, text FROM chunks
        WHERE lang = %s AND (path = %s OR path LIKE %s) ORDER BY position
        """,
        (lang, path, _under(path)),
    ).fetchall()
    if not rows:
        return f"{path}: no such path", []
    parts = [rows[0]["heading"]]
    size = 0
    paths = []
    for row in rows:
        if size + len(row["text"]) > READ_LIMIT and paths:
            parts.append(f"[... truncated, read a narrower path below {path}]")
            break
        parts.append(f"[{row['path']}]\n{row['text']}")
        paths.append(row["path"])
        size += len(row["text"])
    return "\n\n".join(parts), paths
