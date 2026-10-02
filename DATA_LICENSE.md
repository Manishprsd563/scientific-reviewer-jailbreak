# Data license and third-party notice

This repository accompanies the paper *When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection*. Its contents fall under three licensing regimes.

## 1. Code: MIT
All notebooks and source code (`*.ipynb` and any scripts) are released under the MIT License. See [`LICENSE`](LICENSE).

## 2. Our data: CC BY 4.0
The following material was created by the authors. It is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):

- the 15 adversarial strategy prompts in `datasets/prompt_strategies/`;
- the reviewer system prompts and the evaluation rubric embedded in the notebooks;
- all generated model responses, parsed scores, and derived results in `outputs/`.

## 3. Third-party papers: original licenses apply
The PDFs in `datasets/` (other than `prompt_strategies/`) are third-party works. **We do not relicense them.** They remain under their original terms:

| Folder | Source | License |
|---|---|---|
| `poster_accept/`, `spotlight_accept/`, `rejected_papers_*/` | Public ICLR 2025 submissions on [OpenReview](https://openreview.net/group?id=ICLR.cc/2025/Conference) | CC BY 4.0, as released by OpenReview; credit belongs to the original authors |
| `ACL 2017/`, `EMNLP 2018/` | [ACL Anthology](https://aclanthology.org/) | CC BY 4.0 |
| `template_papers/` | Publicly distributed author templates and sample documents from publishers and venues (e.g., IEEE, ACM, Springer, AIAA, AAAI, CEUR) | The respective publisher's terms |

All credit for these papers belongs to their original authors, who are listed in each PDF and on the paper's source page. We redistribute the files only so that the experiments can be reproduced. If you are an author or rights holder and want a file removed, please open an issue and we will remove it.

## Intended use
The adversarial prompts exist to measure and defend against prompt-injection attacks on LLM-based reviewing. **Do not use them to manipulate any real peer-review process.** Doing so may violate venue policies and research-integrity rules, and the authors do not endorse it.

Model weights are not distributed here. The evaluated models are used under their own licenses and terms of service.
