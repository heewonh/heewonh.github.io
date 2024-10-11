---
layout: post
title: Post Modern Portfolio Theory (PMPT) and its Application to the DB GAPS Competition
---

This was a follow-up project to the [Modern Portfolio Theory (MPT) and its Application to the DB GAPS Competition]({{ site.baseurl }}/_posts/2023-7-20-MPT-DBGAPS). After achieving promising results using a MPT-based model, I became curious as to whether using a PMPT-based model would outperform the MPT-based model. 

I was fascinated by the idea behind PMPT (using downside risk or downside deviation as a measure of risk instead of variance) as it seemed more intuitive and well-aligned with investor objectives. After all, it seems more logical and straightforward to say that risk-averse investors aim to minimize 'losses' than to say they aim to minimize 'volatility'. Furthermore, defining risk as variance runs the risk (pun intended) of considering positive deviations from the mean as risky. This would penalize large positive returns under mean-variance optimization and could undermine the validity of the optimization results altogether.

Despite its theoretical soundness, the PMPT-based model failed to consistently outperform the simpler MPT-based model. I encountered lots of difficulty in implementing the model, especially with regard to deriving the lognormal distribution of portfolio returns. I also found that having to arbitrarily choose a MAR (Minimum Acceptable Return) that serves as a boundary between upside and downside risk opens up much room for subjectivity which could be a double-edged sword. Optimizing the selection of MAR could, however, be an interesting area for further study.

For more detail, please refer to the following:
- [Discussion slides]({{ site.baseurl }}/documents/YIG_41기_퀀트팀_Portfolio Optimization.pdf)
- [Code](https://github.com/heewonh/PMPT)