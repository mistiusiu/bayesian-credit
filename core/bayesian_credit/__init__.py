"""bayesian_credit — hierarchical Bayesian sequential decision-making for credit risk.

Layers (SPEC §0):
  data        -> schema / leakage guard / loader / synthetic / outcomes / arms / segmentation / panel
  models      -> Two-Stage Beta-Beta (hierarchical Beta-Bernoulli + Beta-MoM recovery), MCTM
  rewards     -> compound risk-adjusted reward R_t = Revenue_t - E[Loss]_t - Funding_t
  bandits     -> Gaussian / contextual Thompson Sampling and baselines
  monitoring  -> PSI Sentinel + circuit breaker
  shadow      -> counterfactual shadow lending engine
  simulation  -> event-sourced, resumable replay of the Lending Club stream
  evaluation  -> Table 11 metrics
"""

__version__ = "0.1.0"
