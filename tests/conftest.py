"""Keep the suite offline: no traces sent, no reranker called.

Set before any project module is imported, and environment variables win over
.env (python-dotenv never overrides), so a developer's real Langfuse and Cohere
keys — or a LangSmith flag left on — cannot turn a unit test into a network call.
"""

import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["RAG_RERANK"] = "false"
