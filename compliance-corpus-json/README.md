# Compliance corpus — JSON

Parsed regulatory corpus, ready to index. **27 documents · 25 frameworks ·
1,437 sections · 2,602 chunks.** JSON only, no code.

Layout mirrors the AuditZero repo: `frameworks/<sector>/<FRAMEWORK_ID>/`.

```
compliance-corpus-json/
├── all_chunks.json                  ← 2,602 chunks. Index this.
├── index.json                       ← manifest of all 27 documents
├── qa_report.json                   ← per-document parse quality
├── validation.json                  ← assertion results (11/12 pass)
└── frameworks/
    ├── corporate/   COMPANIES_ACT_2013/  LABOUR_CODES_2020/  ...
    ├── financial/   RBI_KYC_MASTER_DIRECTIONS/  PCI_DSS_V4/  ...
    ├── global/      GDPR_2018/  ISO_27001_2022/
    └── universal/   DPDP_ACT_2023/  IT_ACT_2000/  ...
```

Each framework folder contains:

| File | Contents |
| --- | --- |
| `metadata.json` | Unchanged from the source repo — regulator, effective date, source URL, SHA-256 |
| `<DOC>.chunks.json` | JSON **array** of retrieval chunks |
| `<DOC>.document.json` | Whole sections + parser audit trail |

Two frameworks ship two PDFs each (`PCI_DSS_V4`, `SEBI_CYBERSECURITY_FRAMEWORK_2024`),
so those folders hold two pairs, suffixed with the PDF stem.

## Loading

```python
import json
chunks = json.load(open("all_chunks.json"))       # 2,602 dicts

texts  = [c["embed_text"] for c in chunks]        # embed THIS
meta   = [{k: c[k] for k in ("chunk_id", "framework_id", "sector",
                             "breadcrumb", "page_start")} for c in chunks]
```

Or one framework at a time:

```python
kyc = json.load(open("frameworks/financial/RBI_KYC_MASTER_DIRECTIONS/"
                     "RBI_KYC_MASTER_DIRECTIONS.chunks.json"))
```

## Chunk fields

| Field | Use |
| --- | --- |
| `embed_text` | **Embed this.** `text` prefixed with framework/regulator/date + breadcrumb |
| `text` | **Display this.** Clean clause text |
| `chunk_id`, `section_id` | Cite these |
| `framework_id`, `sector` | Vector-store filter keys |
| `framework_name`, `regulator`, `effective_date`, `document_status` | Payload metadata |
| `section_number`, `section_title`, `section_path`, `chapter` | Position in the document |
| `breadcrumb` | Human-readable citation location |
| `page_start`, `page_end` | Page provenance back to the PDF |
| `part_index` / `part_total` | Which slice of a split section this is |
| `source_url`, `source_pdf` | Origin |
| `char_count`, `token_estimate` | Sizing |

Embedding the breadcrumb encodes *where* a clause sits, not only what it says.
Always embed `embed_text`; always show `text`.

## Quality

11 of 12 validation checks pass across all 2,602 chunks (see `validation.json`):
schema completeness, `chunk_id` uniqueness, no truncation spike, no fused
footnote markers, no spliced footnote bodies, no control characters, size
budget, page ranges, provenance. The one failure is a false positive — an RBI KYC
chunk ending on the abbreviation `Sr. PPS to HS`.

Six documents had no text layer and were recovered by OCR at 92–95% mean
confidence: all three SEBI frameworks, the CSCRF extension notice, Consumer
Protection E-commerce Rules, and the SEBI Algo Trading circular. `ocr_used` in
each `.document.json` and in `index.json` records which.

**17 frameworks parse with no flags.** Review these before indexing
(`qa_report.json` has the detail):

| Flag | Frameworks | Meaning |
| --- | --- | --- |
| `WEAK_STRUCTURE_DETECTION` | PCI PTS POI supplementary | Section numbers unreliable — don't cite them. Text is fine |
| `MUCH_TEXT_OUTSIDE_SECTIONS` | GDPR_2018 | Expected: 173 recitals precede Article 1 |
| `MANY_SHORT_CHUNKS` | TELEMEDICINE, IRDAI_AML_CFT, RBI_ACCOUNT_AGGREGATOR, RBI_CYBERSECURITY, SEBI_CSCRF notice | Definition lists — short but coherent |
| `CHUNKS_RUN_LONG` | IRDAI_CYBER_SECURITY, RBI_IT_OUTSOURCING, SEBI_CYBERSECURITY_2024 | Chunks sit near the 3,200-char ceiling |

OCR'd text will contain character-level errors, particularly inside tables and
multi-column layouts. Every known defect is either fixed or flagged above —
nothing fails silently.

## Suggested starter corpus

Ten frameworks, all clean, three sectors — enough for a framework filter to
matter and for DPDP-vs-GDPR cross-framework questions:

RBI_KYC_MASTER_DIRECTIONS · RBI_DIGITAL_LENDING_2025 · PMLA_AML_CFT ·
DPDP_ACT_2023 · DPDP_RULES_2025 · IT_ACT_2000 · IT_RULES_2011 ·
CERT_IN_DIRECTIONS_2022 · ISO_27001_2022 · PCI_DSS_V4
