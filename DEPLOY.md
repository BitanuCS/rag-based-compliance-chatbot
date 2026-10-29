# Deploying to Streamlit Community Cloud

The app is a RAG chat over a Chroma index built from the source PDFs by
`pdf_to_chromadb.py` — OCR of the scanned documents, then Cohere embeddings (about 45
API calls on a free trial key). A Community Cloud container starts with an empty disk on
every cold start, so the index cannot be built there — it is built locally,
published to a Hugging Face dataset repo, and downloaded once per container by
`vectorstore.ensure_index()`.

## One-time setup

### 1. Publish the index

From a working local checkout with a built `chroma_db/`:

```bash
export HF_TOKEN=hf_...          # needs *write* permission
python pdf_to_chromadb.py push-index --public your-name/compliance-chroma
```

Re-run it after any rebuild — the deployment keeps serving the old index until
you do. Pushing code to GitHub does **not** publish the index; they ship
separately. The upload mirrors `chroma_db/`, deleting files the rebuild no
longer has, so stale segment folders do not ride along.

`--public` matters. The index is embeddings and text of published regulation, so
there is nothing to protect, and a public repo is readable with no credentials at
all. A private one means the deployed app needs a token that can see it, and the
Hub reports "no access" and "does not exist" as the same 404 — a token mismatch
there is genuinely hard to tell apart from a missing repo.

To open up a repo that was already pushed as private:

```python
from huggingface_hub import HfApi
HfApi(token="hf_<write-token>").update_repo_settings(
    repo_id="your-name/compliance-chroma", repo_type="dataset", private=False
)
```

### 2. Deploy the app

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `app.py` on the `main` branch.
3. **Under Advanced settings, set the Python version to 3.11.** This is the one
   setting that cannot be committed to the repo, and getting it wrong is a hard
   failure — see below.
4. Under **Advanced settings → Secrets**, paste the contents of
   `.streamlit/secrets.toml.example` with real values filled in. At minimum:

   ```toml
   HF_TOKEN = "hf_..."                                  # read permission is enough here
   HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
   CHROMA_INDEX_REPO = "your-name/compliance-chroma"
   COHERE_API_KEY = "..."                               # embeds each question, reranks
   EMBED_MODEL = "embed-v4.0"                           # must match the index
   EMBED_DIM = "1536"
   LANGFUSE_PUBLIC_KEY = "pk-lf-..."                    # tracing; optional
   LANGFUSE_SECRET_KEY = "sk-lf-..."
   ```

   With a public index repo the token is not needed to fetch the index, but it is
   still required for the chat itself — `chat.py` calls a Hugging Face inference
   endpoint. Rotating the token means updating it here too, or the index loads
   and every question fails.

   `EMBED_MODEL` and `EMBED_DIM` must be the ones the index was built with. The
   index records them, and the app refuses to start against a mismatch rather
   than returning unrelated passages. Without the Langfuse keys tracing is off,
   which does not stop the app.

   `_bootstrap.load_secrets_into_env()` copies these into `os.environ` before any
   module reads its configuration, so the existing `os.getenv` calls keep working
   unchanged.

The first cold start installs dependencies and downloads the index. There are
no model weights to pull: embeddings and reranking are Cohere API
calls. Later starts reuse the container's disk and are faster.

## What was changed for deployment

- **`.python-version`** records 3.11 for local tooling (pyenv, uv). It does
  **not** control Streamlit Community Cloud — see the warning below.
- **`requirements.txt`** is now runtime-only. `langchain-huggingface` was missing
  and is imported by both `chat.py` and `vectorstore.py`, so the app could not
  have started without it. `langgraph`, `langchain-community`, `langchain`,
  `rank-bm25`, `ragas`, `datasets`, `google-genai` and `groq` are not imported by
  any source file; `ragas`, `datasets` and `groq` moved to
  `requirements-dev.txt` and the rest were dropped.
- **No torch.** Embeddings moved from local BGE to the Cohere API, as did
  reranking, so torch, transformers and sentence-transformers left
  `requirements.txt` — roughly 1GB less to install and hold in memory.
- **OpenTelemetry is pinned.** langfuse moves the OTel exporters forward and
  chromadb 0.5.23 breaks at import if its gRPC exporter drifts from them; the two
  pins in `requirements.txt` keep them on one release.
- **`pysqlite3-binary`** is installed on Linux, and `_bootstrap.py` shadows the
  stdlib `sqlite3` with it *only* when the host sqlite is older than 3.35, which
  is chromadb's floor.
- **`_bootstrap.py`** also bridges `st.secrets` into `os.environ`, keeping
  Streamlit out of `vectorstore.py`, `store.py` and `chat.py`.

## The Python version must be set in the dashboard

Streamlit Community Cloud **ignores `.python-version`**. It picks the interpreter
from the app's own Advanced settings and defaults to a current release, which at
time of writing is 3.14. This stack cannot run there:

- `langchain-chroma` requires `<3.13`
- `numpy==1.26.4` publishes no wheels for 3.13+
- `pysqlite3-binary` has no cp314 ABI wheel

The failure surfaces as a dependency resolution error, not a version error, so it
reads as a bad pin:

```
Using Python 3.14.7 environment at /home/adminuser/venv
ERROR: Could not find a version that satisfies the requirement
       pysqlite3-binary==0.5.4 (from versions: 0.5.4.post2)
```

That message is misleading — `0.5.4` exists and ships a cp311 wheel. pip is
reporting what is installable *on 3.14*, where the answer is nothing.

**Fix:** set Python to 3.11 in Advanced settings. 3.12 also resolves, but 3.11 is
what the pins are tested against. Streamlit only offers this choice when an app is
first created — if the field is greyed out on the existing app, delete it and
deploy again from the same repo, choosing 3.11 this time. Deleting the app does
not touch the GitHub repo or the published index.

## When it fails

Each of these hides the next, so they tend to arrive in this order:

| Error in the app | Cause | Fix |
| --- | --- | --- |
| `RuntimeError: No COHERE_API_KEY` at page load | Secret missing | Add `COHERE_API_KEY` to Secrets |
| `EmbeddingModelMismatch` at page load | Published index is from another embedding model | `push-index` from a current local build, then **Reboot** |
| `UnauthorizedError` on the first question | Cohere rejected the key (the page loads without one) | Paste the working key again, no spaces inside the quotes |
| `HfHubHTTPError` on a question that retrieved passages | Hugging Face 402 (credits used up) or 401 (token) | Check the logs: 402 needs credits or another provider |
| Every question refused | An old `RAG_MAX_DISTANCE = "0.45"` in Secrets | Delete it; the default is 0.62 |

Only two secrets are real credentials: `HF_TOKEN` and `COHERE_API_KEY`.
`CHROMA_INDEX_REPO` is a repo name; the Langfuse/LangSmith keys are optional;
every other setting has the right default in code. Delete a line rather than
leaving it empty — `EMBED_DIM = ""` crashes, a missing line uses the default.
After any secrets change, **Reboot** the app.

## Known limitations

**Chat history does not survive restarts.** `store.py` writes `chat_history.db`
to local disk, and the Community Cloud container is ephemeral — the database is
recreated empty whenever the app restarts or wakes from sleep. Conversations are
lost. Fixing this means pointing `store.py` at a hosted Postgres or Turso.

**Every question costs two Cohere calls before the LLM:** embedding the question
and reranking. A trial key allows 1,000 calls a month across both, so a free
deployment answers roughly 450 questions a month. A public deployment needs a
production key; if only the rerank call fails, retrieval falls back to vector
order rather than failing.
