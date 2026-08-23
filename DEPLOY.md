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
python ingest.py push-index your-name/compliance-chroma
```

This creates a private dataset repo and uploads the index. Re-run it after any
rebuild — the deployment keeps serving the old index until you do.

### 2. Deploy the app

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `app.py` on the `main` branch.
3. Under **Advanced settings → Secrets**, paste the contents of
   `.streamlit/secrets.toml.example` with real values filled in. At minimum:

   ```toml
   HF_TOKEN = "hf_..."                                  # read permission is enough here
   HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
   CHROMA_INDEX_REPO = "your-name/compliance-chroma"
   ```

   `_bootstrap.load_secrets_into_env()` copies these into `os.environ` before any
   module reads its configuration, so the existing `os.getenv` calls keep working
   unchanged.

The first cold start installs dependencies, downloads the ~99MB index, and pulls
the `bge-base-en-v1.5` embedding weights (~440MB). Expect several minutes. Later
starts reuse the container's disk and are much faster.

## What was changed for deployment

- **`.python-version`** pins 3.11. `chromadb==0.5.23` and `numpy==1.26.4` do not
  build on 3.13, which is what the platform would otherwise pick.
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
