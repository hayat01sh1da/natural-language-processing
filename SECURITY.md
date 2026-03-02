## Supported Versions

- Only the latest code on `master` is supported. Run `bash run_unittests.sh` using the environment described in the **Ecosystem & Compatibility** section before submitting a report.
- Derived notebooks or experiments that pin older dependencies are considered out of scope unless they reproduce on the current stack.

## Ecosystem & Compatibility

| Component / Library      | Version(s) / Tooling               | Notes |
| ------------------------ | ---------------------------------- | ----- |
| OS baseline              | WSL (Ubuntu 24.04.3 LTS)           | Matches the setup guide. |
| Python runtime           | CPython 3.14.3 (`.python-version`) | Managed with pyenv. |
| Core Python packages     | `numpy`, `pandas`, `scikit-learn`, `scipy`, `gensim`, `keras`, `matplotlib` | Install via `pip install -r requirements.txt` plus documentation references. |
| NLP tooling              | MeCab + ipadic-NEologd, `lzma`, IPA fonts | Follow README instructions for installation prior to running tokenizers. |

## Backward Compatibility

- Code samples are tested on Python 3.14.x. We keep APIs and CLI flags stable within that runtime line; breaking tokenizer changes are documented in the release notes.
- Older Python versions or MeCab dictionary builds are unsupported, and we do not backport fixes to them.

## Reporting a Vulnerability

Please report issues privately via **GitHub Security Advisory** (preferred) — open through the repository’s **Security → Report a vulnerability** workflow.

Acknowledgement occurs and status updates follow as soon as possible.  
After remediation we publish guidance alongside required dependency updates.
