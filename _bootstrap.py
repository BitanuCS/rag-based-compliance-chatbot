"""Deployment-environment fixes that must run before anything else imports.

app.py imports this first, on purpose. Both fixes below have to land before
chromadb is imported and before vectorstore.py/chat.py read their configuration
at module scope, which happens as soon as `import rag` runs.

Importing this locally is a no-op: neither branch fires on a dev machine with a
current sqlite and a .env file.
"""

import os
import sqlite3
import sys

# chromadb requires sqlite >= 3.35 and refuses to start below it. Some Linux
# images still ship 3.31, where the only fix is to shadow the stdlib module with
# the statically-linked build from pysqlite3-binary. Guarded on the actual
# version rather than applied blindly, so an up-to-date host keeps using its own
# sqlite and this stays a no-op.
if sqlite3.sqlite_version_info < (3, 35, 0):
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass  # Nothing better to do; chromadb will raise its own clear error.


def load_secrets_into_env():
    """Copy Streamlit secrets into os.environ.

    Every module here reads configuration with os.getenv, because they are meant
    to stay usable from the CLI and from a future API layer with no Streamlit in
    the picture. On Streamlit Community Cloud there is no .env — configuration
    arrives as st.secrets — so it is bridged across here rather than by teaching
    each module about Streamlit.

    setdefault, not assignment: a real environment variable is the more specific
    signal and keeps working for local runs and for `streamlit run` with a .env.
    Only top-level scalars are bridged; a nested section has no env equivalent.
    """
    try:
        import streamlit as st

        # st.secrets is lazy: it parses on first access, so the read has to
        # happen inside the guard. With no secrets.toml that raises
        # StreamlitSecretNotFoundError, which is the normal local case.
        items = list(st.secrets.items())
    except Exception:
        return

    for key, value in items:
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))
