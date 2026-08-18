"""Research tooling: measuring whether an edge exists, separately from trading it.

Deliberately separate from `src/strategy`. The strategy answers "should I trade
this?" under a policy; research answers "does a tradeable dislocation exist at all,
at what size, and how often?" -- which is the question that has to be settled first
and which the trading path is the wrong shape for.
"""
