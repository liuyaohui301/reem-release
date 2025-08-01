# ReeM: Hierarchical Reinforcement Learning Empowers Dynamic Model Ensemble for HVAC Model Predictive Control
Model Predictive Control (**MPC**) has emerging as a highly effective strategy for Optimizing Heating, Ventilation, and Air Conditioning (**HVAC**) systems. MPC's performance heavily relies on accurate building thermodynamics models, which traditionally suffer from costly  development and limited adaptability factors.

![MPC-based HVAC control optimization process](pic/MPC-based%20HVAC%20control%20optimization%20process.jpg)

The goal of this work is to show that the effectiveness and efficiency of thermodynamics modeling can be greatly enhanced using Reinforcement Learning (**RL**).

To this end, we propose a novel Hierarchical Reinforcement Learning (HRL)-based framework, `ReeM`, that learns dynamic model ensemble policy. `ReeM` intelligently selects the most suitable models from a pre-prepared pool and assigns optimal weights for real-time thermal dynamics prediction within each MPC optimization horizon. This process solely from the observed data stream without requiring explicit knowledge of the underlying physics in this domain.

![reem_implementation](pic/reem_implementation.jpg)

Specifically, `ReeM` formulates the ensemble task as a two-level decision-making process: the high-level agent determines whether a model should be included in the ensemble; the low-level agent assigns weights to each selected model. Additionally, a data representation module extracts features from observed system states to enable effective actions by the agents.

Our on-site experiments confirm that `ReeM` can achieve 47.5\% improvement in modeling prediction accuracy, and a significant 8.7\% energy saving in real HVAC control.

![reem on-site deployment](pic/reem%20on-site%20deployment.jpg)