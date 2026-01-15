# Security Policy

## Supported Versions

- Only the latest code on `master` is supported. Run `bash run_unittests.sh`
	against the versions below before submitting a report.
- Derived notebooks or experiments that pin older dependencies are considered
	out of scope unless they reproduce on the current stack.

## Ecosystem & Compatibility

| Component / Library      | Version(s) / Tooling               | Notes |
| ------------------------ | ---------------------------------- | ----- |
| OS baseline              | WSL (Ubuntu 24.04.3 LTS)           | Matches the setup guide. |
| Python runtime           | CPython 3.14.2 (`.python-version`) | Managed with pyenv. |
| Core Python packages     | `numpy`, `pandas`, `scikit-learn`, `scipy`, `gensim`, `keras`, `matplotlib` | Install via `pip install -r requirements.txt` plus documentation references. |
| NLP tooling              | MeCab + ipadic-NEologd, `lzma`, IPA fonts | Follow README instructions for installation prior to running tokenizers. |

## Backward Compatibility

- Code samples are tested on Python 3.14.x. We keep APIs and CLI flags stable
	within that runtime line; breaking tokenizer changes are documented in the
	release notes.
- Older Python versions or MeCab dictionary builds are unsupported, and we do
	not backport fixes to them.

## Reporting a Vulnerability

Please disclose vulnerabilities privately:

1. Use GitHub **Security → Report a vulnerability** to open a private advisory
	 (preferred). Include dataset links and scripts/notebooks that reproduce the
	 issue.
2. Or email `security@project.org` with detailed steps, affected components,
	 and suggested mitigations.

We acknowledge within **3 business days** and share updates at least every **7
business days** during investigation.
