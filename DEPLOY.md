# Deploying to Streamlit Community Cloud

The app is a RAG chat over a ~99MB Chroma index that takes about 35 minutes to
build. A Community Cloud container starts with an empty disk on every cold start,
so the index cannot be built there — it is built locally, published to a Hugging
Face dataset repo, and downloaded once per container by
`vectorstore.ensure_index()`.

## One-time setup

### 1. Publish the index

From a working local checkout with a built `chroma_db/`:

```bash
export HF_TOKEN=hf_...          # needs *write* permission
python ingest.py push-index --public your-name/compliance-chroma
```

Re-run it after any rebuild — the deployment keeps serving the old index until
you do.

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
   ```

   With a public index repo the token is not needed to fetch the index, but it is
   still required for the chat itself — `chat.py` calls a Hugging Face inference
   endpoint. Rotating the token means updating it here too, or the index loads
   and every question fails.

   `_bootstrap.load_secrets_into_env()` copies these into `os.environ` before any
   module reads its configuration, so the existing `os.getenv` calls keep working
   unchanged.

The first cold start installs dependencies, downloads the ~99MB index, and pulls
the `bge-base-en-v1.5` embedding weights (~440MB). Expect several minutes. Later
starts reuse the container's disk and are much faster.

## What was changed for deployment

- **`.python-version`** records 3.11 for local tooling (pyenv, uv). It does
  **not** control Streamlit Community Cloud — see the warning below.
- **`requirements.txt`** is now runtime-only. `langchain-huggingface` was missing
  and is imported by both `chat.py` and `vectorstore.py`, so the app could not
  have started without it. `langgraph`, `langchain-community`, `langchain`,
  `rank-bm25`, `ragas`, `datasets`, `google-genai` and `groq` are not imported by
  any source file and moved to `requirements-dev.txt`.
- **torch comes from the CPU wheel index.** The default PyPI wheel bundles CUDA
  and unpacks to ~2.5GB, which does not fit the container.
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

## Known limitations

**Chat history does not survive restarts.** `store.py` writes `chat_history.db`
to local disk, and the Community Cloud container is ephemeral — the database is
recreated empty whenever the app restarts or wakes from sleep. Conversations are
lost. Fixing this means pointing `store.py` at a hosted Postgres or Turso.

**Memory is tight.** torch, transformers and the loaded embedding model sit in
RAM alongside the Chroma HNSW index, against roughly 2.7GB on the free tier. If
the app is killed on startup, the next thing to cut is local embeddings — call
the Hugging Face inference API for query embeddings instead, which drops torch
and transformers entirely.

**`_device()` returns CPU.** The MPS branch in `vectorstore.py` is for Apple
Silicon dev machines and never fires on the deployed Linux container, so query
embedding is slower there.
