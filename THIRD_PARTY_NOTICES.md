# Third-party notices

## AutoMPO

This repository includes a lightweight vendored copy of the finite-state-automaton
AutoMPO implementation by **Hao-Kai Zhang**:

- Upstream: <https://github.com/Haokai-Zhang/AutoMPO>
- Upstream commit: `b786858a28865f926a27e16df89e4d3c2515a0f9`
- Original author notice: `Hao-Kai Zhang <zhk20@tsinghua.mails.edu.cn>`

The files `class_fsa.py`, `class_named_data.py`, and `mpo_gadgets.py` match that
upstream commit apart from package-relative imports. The supplied `opr_pool.py`
adds the `sigmaX`, `sigmaY`, and `sigmaZ` aliases to the upstream operator pool.

The upstream repository did not declare a software license when this copy was
prepared on 2026-07-16. Copyright and redistribution rights for these files remain
with the original author; they are not represented here as original Apache-2.0
TNVD code. The TNVD repository maintainer states that this implementation was
provided directly by the author. Downstream redistributors should confirm their
intended use with the original author.
