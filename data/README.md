# Data

Nothing is downloaded. `build_domains.py` builds three text files from what is installed on the machine; the built files are committed so results do not depend on the Python or Node version.

| File | Role | Source | Licence |
|---|---|---|---|
| `code.txt` | New domain, adapter 1 | Python standard-library source files (CPython 3.13) with module, class and function docstrings removed | PSF-2.0 |
| `js.txt` | New domain, adapter 2 (the adapter-swapping experiment) | JavaScript source files of four command-line packages found in the global npm folder: ESLint 10.1.0 (`lib/`), nodemon, http-server, serve | MIT (copyright remains with each package's authors; the files carry their own headers) |
| `old_domain_val.txt` | The base model's own domain, used only to measure forgetting | The 46 validation documents of the base corpus of [transformer-from-scratch](https://github.com/enxo7899/transformer-from-scratch): Python documentation and docstrings | PSF-2.0 |

**Credit.** The Python standard library and documentation are copyright of the Python Software Foundation and contributors, distributed under the PSF License Version 2 (https://docs.python.org/3/license.html). The JavaScript files are copyright of the ESLint, nodemon, http-server and serve authors under the MIT licence. Text is quoted unchanged apart from being reduced to plain ASCII, tabs to four spaces, runs of blank lines collapsed, and truncation to the first 6,000 characters per file. Files that name an AI vendor are left out.

**Shape.** Documents are separated by a line holding only `<<<DOC>>>`. Each new-domain file holds about 1.3 to 1.5 million characters, one document per source file (up to 6,000 characters). Repetitive files (data tables, generated code, minified bundles) are dropped.

**Split.** `lora_scratch.finetune.load_domain` shuffles whole documents with a fixed seed and holds out 10% of characters as validation, then 6% of what is left as a dev split used only to choose learning rates. No document appears in two parts (a test checks this). The tokenizer is the base model's, unchanged, for every run: it was trained on documentation, so it is inefficient on code, which is part of what makes the new domain hard.

**Rebuild.** `python3 data/build_domains.py` (add `--base-corpus path/to/corpus.txt` to regenerate `old_domain_val.txt` from a checkout of the base repo).
