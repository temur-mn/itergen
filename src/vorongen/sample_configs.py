"""
Sample YAML configurations stored as strings.
"""

import copy
from typing import Any

import yaml

CONFIG_BINARY = """
metadata:
  name: "binary_only_demo"
  version: "1.0"

columns:
  - column_id: "loyalty"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "bernoulli"
      probabilities: { true_prob: 0.4, false_prob: 0.6 }

  - column_id: "discount"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["loyalty"]
      conditional_probs:
        "loyalty=1": { true_prob: 0.6, false_prob: 0.4 }
        "loyalty=0": { true_prob: 0.2, false_prob: 0.8 }

  - column_id: "online"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["discount"]
      conditional_probs:
        "discount=1": { true_prob: 0.2, false_prob: 0.8 }
        "discount=0": { true_prob: 0.1, false_prob: 0.9 }
"""

CONFIG_CATEGORICAL = """
metadata:
  name: "categorical_only_demo"
  version: "1.0"

columns:
  - column_id: "device"
    values:
      categories: ["mobile", "desktop", "tablet"]
    distribution:
      type: "categorical"
      probabilities: { mobile: 0.5, desktop: 0.35, tablet: 0.15 }

  - column_id: "region"
    values:
      categories: ["na", "emea", "apac"]
    distribution:
      type: "categorical"
      depend_on: ["device"]
      conditional_probs:
        "device=mobile": { na: 0.4, emea: 0.35, apac: 0.25 }
        "device=desktop": { na: 0.5, emea: 0.3, apac: 0.2 }
        "device=tablet": { na: 0.3, emea: 0.3, apac: 0.4 }
"""

CONFIG_BINARY_CATEGORICAL_LARGE = """
metadata:
  name: "binary_categorical_large_demo"
  version: "1.0"

columns:
  - column_id: "loyalty"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "bernoulli"
      probabilities: { true_prob: 0.45, false_prob: 0.55 }

  - column_id: "discount"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["loyalty"]
      conditional_probs:
        "loyalty=1": { true_prob: 0.55, false_prob: 0.45 }
        "loyalty=0": { true_prob: 0.25, false_prob: 0.75 }

  - column_id: "channel"
    values:
      categories: ["web", "app", "store"]
    distribution:
      type: "categorical"
      probabilities: { web: 0.45, app: 0.35, store: 0.20 }

  - column_id: "device"
    values:
      categories: ["mobile", "desktop", "tablet"]
    distribution:
      type: "categorical"
      depend_on: ["channel"]
      conditional_probs:
        "channel=web": { mobile: 0.35, desktop: 0.50, tablet: 0.15 }
        "channel=app": { mobile: 0.70, desktop: 0.20, tablet: 0.10 }
        "channel=store": { mobile: 0.30, desktop: 0.45, tablet: 0.25 }

  - column_id: "region"
    values:
      categories: ["na", "emea", "apac"]
    distribution:
      type: "categorical"
      depend_on: ["device", "loyalty"]
      conditional_probs:
        "device=mobile, loyalty=1": { na: 0.40, emea: 0.35, apac: 0.25 }
        "device=desktop, loyalty=1": { na: 0.45, emea: 0.30, apac: 0.25 }
        "device=tablet, loyalty=1": { na: 0.35, emea: 0.30, apac: 0.35 }
        "device=mobile, loyalty=0": { na: 0.30, emea: 0.40, apac: 0.30 }
        "device=desktop, loyalty=0": { na: 0.35, emea: 0.35, apac: 0.30 }
        "device=tablet, loyalty=0": { na: 0.25, emea: 0.35, apac: 0.40 }

  - column_id: "plan_tier"
    values:
      categories: ["basic", "plus", "pro"]
    distribution:
      type: "categorical"
      depend_on: ["loyalty", "channel"]
      conditional_probs:
        "loyalty=1, channel=web": { basic: 0.25, plus: 0.45, pro: 0.30 }
        "loyalty=1, channel=app": { basic: 0.20, plus: 0.40, pro: 0.40 }
        "loyalty=1, channel=store": { basic: 0.30, plus: 0.45, pro: 0.25 }
        "loyalty=0, channel=web": { basic: 0.55, plus: 0.30, pro: 0.15 }
        "loyalty=0, channel=app": { basic: 0.45, plus: 0.35, pro: 0.20 }
        "loyalty=0, channel=store": { basic: 0.50, plus: 0.35, pro: 0.15 }

  - column_id: "auto_pay"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["plan_tier", "loyalty"]
      conditional_probs:
        "plan_tier=basic, loyalty=1": { true_prob: 0.45, false_prob: 0.55 }
        "plan_tier=plus, loyalty=1": { true_prob: 0.55, false_prob: 0.45 }
        "plan_tier=pro, loyalty=1": { true_prob: 0.65, false_prob: 0.35 }
        "plan_tier=basic, loyalty=0": { true_prob: 0.30, false_prob: 0.70 }
        "plan_tier=plus, loyalty=0": { true_prob: 0.38, false_prob: 0.62 }
        "plan_tier=pro, loyalty=0": { true_prob: 0.45, false_prob: 0.55 }

  - column_id: "support_channel"
    values:
      categories: ["chat", "email", "phone"]
    distribution:
      type: "categorical"
      depend_on: ["plan_tier", "channel"]
      conditional_probs:
        "plan_tier=basic, channel=web": { chat: 0.40, email: 0.40, phone: 0.20 }
        "plan_tier=basic, channel=app": { chat: 0.52, email: 0.33, phone: 0.15 }
        "plan_tier=basic, channel=store": { chat: 0.25, email: 0.35, phone: 0.40 }
        "plan_tier=plus, channel=web": { chat: 0.45, email: 0.35, phone: 0.20 }
        "plan_tier=plus, channel=app": { chat: 0.55, email: 0.30, phone: 0.15 }
        "plan_tier=plus, channel=store": { chat: 0.30, email: 0.30, phone: 0.40 }
        "plan_tier=pro, channel=web": { chat: 0.35, email: 0.30, phone: 0.35 }
        "plan_tier=pro, channel=app": { chat: 0.45, email: 0.25, phone: 0.30 }
        "plan_tier=pro, channel=store": { chat: 0.20, email: 0.30, phone: 0.50 }

  - column_id: "priority_support"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["plan_tier", "support_channel"]
      conditional_probs:
        "plan_tier=basic, support_channel=chat": { true_prob: 0.15, false_prob: 0.85 }
        "plan_tier=basic, support_channel=email": { true_prob: 0.10, false_prob: 0.90 }
        "plan_tier=basic, support_channel=phone": { true_prob: 0.20, false_prob: 0.80 }
        "plan_tier=plus, support_channel=chat": { true_prob: 0.25, false_prob: 0.75 }
        "plan_tier=plus, support_channel=email": { true_prob: 0.20, false_prob: 0.80 }
        "plan_tier=plus, support_channel=phone": { true_prob: 0.35, false_prob: 0.65 }
        "plan_tier=pro, support_channel=chat": { true_prob: 0.45, false_prob: 0.55 }
        "plan_tier=pro, support_channel=email": { true_prob: 0.35, false_prob: 0.65 }
        "plan_tier=pro, support_channel=phone": { true_prob: 0.55, false_prob: 0.45 }

  - column_id: "tenure_bucket"
    values:
      categories: ["new", "mid", "long"]
    distribution:
      type: "categorical"
      depend_on: ["auto_pay", "loyalty"]
      conditional_probs:
        "auto_pay=1, loyalty=1": { new: 0.20, mid: 0.40, long: 0.40 }
        "auto_pay=0, loyalty=1": { new: 0.35, mid: 0.45, long: 0.20 }
        "auto_pay=1, loyalty=0": { new: 0.30, mid: 0.45, long: 0.25 }
        "auto_pay=0, loyalty=0": { new: 0.45, mid: 0.40, long: 0.15 }

  - column_id: "payment_method"
    values:
      categories: ["card", "bank", "wallet"]
    distribution:
      type: "categorical"
      depend_on: ["channel", "auto_pay"]
      conditional_probs:
        "channel=web, auto_pay=1": { card: 0.55, bank: 0.30, wallet: 0.15 }
        "channel=web, auto_pay=0": { card: 0.60, bank: 0.25, wallet: 0.15 }
        "channel=app, auto_pay=1": { card: 0.30, bank: 0.20, wallet: 0.50 }
        "channel=app, auto_pay=0": { card: 0.35, bank: 0.25, wallet: 0.40 }
        "channel=store, auto_pay=1": { card: 0.45, bank: 0.40, wallet: 0.15 }
        "channel=store, auto_pay=0": { card: 0.50, bank: 0.35, wallet: 0.15 }

  - column_id: "churn_risk"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["plan_tier", "priority_support", "discount"]
      conditional_probs:
        "plan_tier=basic, priority_support=1, discount=1": { true_prob: 0.25, false_prob: 0.75 }
        "plan_tier=basic, priority_support=1, discount=0": { true_prob: 0.35, false_prob: 0.65 }
        "plan_tier=basic, priority_support=0, discount=1": { true_prob: 0.30, false_prob: 0.70 }
        "plan_tier=basic, priority_support=0, discount=0": { true_prob: 0.45, false_prob: 0.55 }
        "plan_tier=plus, priority_support=1, discount=1": { true_prob: 0.18, false_prob: 0.82 }
        "plan_tier=plus, priority_support=1, discount=0": { true_prob: 0.25, false_prob: 0.75 }
        "plan_tier=plus, priority_support=0, discount=1": { true_prob: 0.22, false_prob: 0.78 }
        "plan_tier=plus, priority_support=0, discount=0": { true_prob: 0.32, false_prob: 0.68 }
        "plan_tier=pro, priority_support=1, discount=1": { true_prob: 0.12, false_prob: 0.88 }
        "plan_tier=pro, priority_support=1, discount=0": { true_prob: 0.18, false_prob: 0.82 }
        "plan_tier=pro, priority_support=0, discount=1": { true_prob: 0.15, false_prob: 0.85 }
        "plan_tier=pro, priority_support=0, discount=0": { true_prob: 0.25, false_prob: 0.75 }
"""

CONFIG_CONTINUOUS = """
metadata:
  name: "continuous_only_demo"
  version: "1.0"
  continuous_bin_conflict_mode: error
  continuous_bin_mean_error_max: 0.12
  continuous_bin_max_error_max: 0.30
  continuous_violation_rate_max: 0.02
  continuous_mean_violation_max: 1.0
  continuous_max_violation_max: 5.0

columns:
  - column_id: "spend"
    distribution:
      type: "continuous"
      targets: { mean: 120.0, std: 30.0, min: 20.0, max: 240.0 }
      conditioning_bins:
        edges: [20.0, 60.0, 95.0, 125.0, 155.0, 190.0, 240.0]
        labels: ["vlow", "low", "mid", "high", "vhigh", "top"]

  - column_id: "session_length"
    distribution:
      type: "continuous"
      targets: { mean: 12.0, std: 4.0, min: 1.0, max: 40.0 }
      conditioning_bins:
        edges: [1.0, 4.0, 7.0, 10.0, 14.0, 20.0, 40.0]
        labels: ["short", "quick", "mid", "long", "xlong", "extreme"]
"""

CONFIG_MIXED = """
metadata:
  name: "mixed_demo"
  version: "1.0"
  continuous_bin_conflict_mode: error
  continuous_bin_mean_error_max: 0.12
  continuous_bin_max_error_max: 0.30
  continuous_violation_rate_max: 0.03
  continuous_mean_violation_max: 1.5
  continuous_max_violation_max: 8.0

columns:
  - column_id: "loyalty"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "bernoulli"
      probabilities: { true_prob: 0.4, false_prob: 0.6 }

  - column_id: "device"
    values:
      categories: ["mobile", "desktop", "tablet"]
    distribution:
      type: "categorical"
      depend_on: ["loyalty"]
      conditional_probs:
        "loyalty=1": { mobile: 0.6, desktop: 0.3, tablet: 0.1 }
        "loyalty=0": { mobile: 0.4, desktop: 0.4, tablet: 0.2 }

  - column_id: "subscription"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["device"]
      conditional_probs:
        "device=mobile": { true_prob: 0.5, false_prob: 0.5 }
        "device=desktop": { true_prob: 0.3, false_prob: 0.7 }
        "device=tablet": { true_prob: 0.2, false_prob: 0.8 }

  - column_id: "spend"
    distribution:
      type: "continuous"
      targets: { mean: 110.0, std: 25.0, min: 10.0, max: 220.0 }
      conditioning_bins:
        edges: [10.0, 45.0, 75.0, 105.0, 135.0, 170.0, 220.0]
        labels: ["b1", "b2", "b3", "b4", "b5", "b6"]
"""

CONFIG_CONTINUOUS_PARENT_BINS = """
metadata:
  name: "continuous_parent_bins_demo"
  version: "1.0"
  continuous_bin_conflict_mode: error
  continuous_bin_mean_error_max: 0.14
  continuous_bin_max_error_max: 0.32

columns:
  - column_id: "risk_score"
    distribution:
      type: "continuous"
      targets: { mean: 50.0, std: 15.0, min: 0.0, max: 100.0 }
      conditioning_bins:
        edges: [0.0, 35.0, 65.0, 100.0]
        labels: ["low", "mid", "high"]
      bin_probs: { low: 0.27, mid: 0.46, high: 0.27 }

  - column_id: "retained"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["risk_score"]
      conditional_probs:
        "risk_score=low": { true_prob: 0.80, false_prob: 0.20 }
        "risk_score=mid": { true_prob: 0.50, false_prob: 0.50 }
        "risk_score=high": { true_prob: 0.25, false_prob: 0.75 }

  - column_id: "support_cost"
    distribution:
      type: "continuous"
      depend_on: ["risk_score"]
      conditioning_bins:
        edges: [1.0, 8.0, 12.0, 16.0, 22.0, 30.0, 40.0]
        labels: ["vlow", "low", "mid", "high", "vhigh", "extreme"]
      bin_probs: { vlow: 0.10, low: 0.22, mid: 0.28, high: 0.22, vhigh: 0.13, extreme: 0.05 }
      conditional_targets:
        "risk_score=low": { mean: 8.0, std: 2.0, min: 1.0, max: 20.0 }
        "risk_score=mid": { mean: 14.0, std: 3.0, min: 3.0, max: 30.0 }
        "risk_score=high": { mean: 22.0, std: 4.0, min: 6.0, max: 40.0 }
"""

CONFIG_MIXED_LARGE = """
metadata:
  n_rows: 30000
  tolerance: 0.05
  max_attempts: 3
  attempt_workers: 4
  proposal_scoring_mode: incremental
  continuous_bin_conflict_mode: error
  log_level: info
  objective_max: 0.40
  max_error_max: 0.1
  max_column_deviation_max: 0.1
  continuous_bin_mean_error_max: 0.05
  continuous_bin_max_error_max: 0.09
  continuous_violation_rate_max: 0.03
  continuous_mean_violation_max: 0.75
  continuous_max_violation_max: 1.0

advanced:
  enabled: true
  batch_size: 2048
  proposals_per_batch: 16
  max_iters: 120
  patience: 8
  temperature_decay: 0.92
  random_flip_frac: 0.006
  step_size_marginal: 0.22
  step_size_conditional: 0.33
  step_size_continuous_marginal: 0.28
  step_size_continuous_conditional: 0.42
  continuous_dependency_gain: 0.20
  continuous_noise_frac: 0.05
  continuous_edge_guard_frac: 0.04

columns:
  - column_id: "loyalty"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "bernoulli"
      probabilities: { true_prob: 0.45, false_prob: 0.55 }

  - column_id: "discount"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["loyalty"]
      conditional_probs:
        "loyalty=1": { true_prob: 0.55, false_prob: 0.45 }
        "loyalty=0": { true_prob: 0.25, false_prob: 0.75 }

  - column_id: "channel"
    values:
      categories: ["web", "app", "store"]
    distribution:
      type: "categorical"
      probabilities: { web: 0.45, app: 0.35, store: 0.20 }

  - column_id: "device"
    values:
      categories: ["mobile", "desktop", "tablet"]
    distribution:
      type: "categorical"
      depend_on: ["channel"]
      conditional_probs:
        "channel=web": { mobile: 0.35, desktop: 0.50, tablet: 0.15 }
        "channel=app": { mobile: 0.70, desktop: 0.20, tablet: 0.10 }
        "channel=store": { mobile: 0.30, desktop: 0.45, tablet: 0.25 }

  - column_id: "region"
    values:
      categories: ["na", "emea", "apac"]
    distribution:
      type: "categorical"
      depend_on: ["device", "loyalty"]
      conditional_probs:
        "device=mobile, loyalty=1": { na: 0.40, emea: 0.35, apac: 0.25 }
        "device=desktop, loyalty=1": { na: 0.45, emea: 0.30, apac: 0.25 }
        "device=tablet, loyalty=1": { na: 0.35, emea: 0.30, apac: 0.35 }
        "device=mobile, loyalty=0": { na: 0.30, emea: 0.40, apac: 0.30 }
        "device=desktop, loyalty=0": { na: 0.35, emea: 0.35, apac: 0.30 }
        "device=tablet, loyalty=0": { na: 0.25, emea: 0.35, apac: 0.40 }

  - column_id: "plan_tier"
    values:
      categories: ["basic", "plus", "pro"]
    distribution:
      type: "categorical"
      depend_on: ["loyalty", "channel"]
      conditional_probs:
        "loyalty=1, channel=web": { basic: 0.25, plus: 0.45, pro: 0.30 }
        "loyalty=1, channel=app": { basic: 0.20, plus: 0.40, pro: 0.40 }
        "loyalty=1, channel=store": { basic: 0.30, plus: 0.45, pro: 0.25 }
        "loyalty=0, channel=web": { basic: 0.55, plus: 0.30, pro: 0.15 }
        "loyalty=0, channel=app": { basic: 0.45, plus: 0.35, pro: 0.20 }
        "loyalty=0, channel=store": { basic: 0.50, plus: 0.35, pro: 0.15 }

  - column_id: "auto_pay"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["plan_tier", "loyalty"]
      conditional_probs:
        "plan_tier=basic, loyalty=1": { true_prob: 0.45, false_prob: 0.55 }
        "plan_tier=plus, loyalty=1": { true_prob: 0.55, false_prob: 0.45 }
        "plan_tier=pro, loyalty=1": { true_prob: 0.65, false_prob: 0.35 }
        "plan_tier=basic, loyalty=0": { true_prob: 0.30, false_prob: 0.70 }
        "plan_tier=plus, loyalty=0": { true_prob: 0.38, false_prob: 0.62 }
        "plan_tier=pro, loyalty=0": { true_prob: 0.45, false_prob: 0.55 }

  - column_id: "support_channel"
    values:
      categories: ["chat", "email", "phone"]
    distribution:
      type: "categorical"
      depend_on: ["plan_tier", "channel"]
      conditional_probs:
        "plan_tier=basic, channel=web": { chat: 0.40, email: 0.40, phone: 0.20 }
        "plan_tier=basic, channel=app": { chat: 0.52, email: 0.33, phone: 0.15 }
        "plan_tier=basic, channel=store": { chat: 0.25, email: 0.35, phone: 0.40 }
        "plan_tier=plus, channel=web": { chat: 0.45, email: 0.35, phone: 0.20 }
        "plan_tier=plus, channel=app": { chat: 0.55, email: 0.30, phone: 0.15 }
        "plan_tier=plus, channel=store": { chat: 0.30, email: 0.30, phone: 0.40 }
        "plan_tier=pro, channel=web": { chat: 0.35, email: 0.30, phone: 0.35 }
        "plan_tier=pro, channel=app": { chat: 0.45, email: 0.25, phone: 0.30 }
        "plan_tier=pro, channel=store": { chat: 0.20, email: 0.30, phone: 0.50 }

  - column_id: "priority_support"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["plan_tier", "support_channel"]
      conditional_probs:
        "plan_tier=basic, support_channel=chat": { true_prob: 0.15, false_prob: 0.85 }
        "plan_tier=basic, support_channel=email": { true_prob: 0.10, false_prob: 0.90 }
        "plan_tier=basic, support_channel=phone": { true_prob: 0.20, false_prob: 0.80 }
        "plan_tier=plus, support_channel=chat": { true_prob: 0.25, false_prob: 0.75 }
        "plan_tier=plus, support_channel=email": { true_prob: 0.20, false_prob: 0.80 }
        "plan_tier=plus, support_channel=phone": { true_prob: 0.35, false_prob: 0.65 }
        "plan_tier=pro, support_channel=chat": { true_prob: 0.45, false_prob: 0.55 }
        "plan_tier=pro, support_channel=email": { true_prob: 0.35, false_prob: 0.65 }
        "plan_tier=pro, support_channel=phone": { true_prob: 0.55, false_prob: 0.45 }

  - column_id: "tenure_bucket"
    values:
      categories: ["new", "mid", "long"]
    distribution:
      type: "categorical"
      depend_on: ["auto_pay", "loyalty"]
      conditional_probs:
        "auto_pay=1, loyalty=1": { new: 0.20, mid: 0.40, long: 0.40 }
        "auto_pay=0, loyalty=1": { new: 0.35, mid: 0.45, long: 0.20 }
        "auto_pay=1, loyalty=0": { new: 0.30, mid: 0.45, long: 0.25 }
        "auto_pay=0, loyalty=0": { new: 0.45, mid: 0.40, long: 0.15 }

  - column_id: "payment_method"
    values:
      categories: ["card", "bank", "wallet"]
    distribution:
      type: "categorical"
      depend_on: ["channel", "auto_pay"]
      conditional_probs:
        "channel=web, auto_pay=1": { card: 0.55, bank: 0.30, wallet: 0.15 }
        "channel=web, auto_pay=0": { card: 0.60, bank: 0.25, wallet: 0.15 }
        "channel=app, auto_pay=1": { card: 0.30, bank: 0.20, wallet: 0.50 }
        "channel=app, auto_pay=0": { card: 0.35, bank: 0.25, wallet: 0.40 }
        "channel=store, auto_pay=1": { card: 0.45, bank: 0.40, wallet: 0.15 }
        "channel=store, auto_pay=0": { card: 0.50, bank: 0.35, wallet: 0.15 }

  - column_id: "spend"
    distribution:
      type: "continuous"
      targets: { mean: 115.0, std: 28.0, min: 10.0, max: 240.0 }
      conditioning_bins:
        edges: [10.0, 45.0, 75.0, 105.0, 135.0, 170.0, 240.0]
        labels: ["sp1", "sp2", "sp3", "sp4", "sp5", "sp6"]

  - column_id: "session_length"
    distribution:
      type: "continuous"
      targets: { mean: 12.5, std: 4.5, min: 1.0, max: 40.0 }
      conditioning_bins:
        edges: [1.0, 4.0, 7.0, 10.0, 14.0, 20.0, 40.0]
        labels: ["sl1", "sl2", "sl3", "sl4", "sl5", "sl6"]

  - column_id: "support_cost"
    distribution:
      type: "continuous"
      depend_on: ["support_channel", "priority_support"]
      conditioning_bins:
        edges: [1.0, 8.0, 14.0, 20.0, 30.0, 45.0, 90.0]
        labels: ["c1", "c2", "c3", "c4", "c5", "c6"]
      bin_probs: { c1: 0.12, c2: 0.24, c3: 0.26, c4: 0.20, c5: 0.13, c6: 0.05 }
      conditional_targets:
        "support_channel=chat, priority_support=1": { mean: 18.0, std: 5.0, min: 2.0, max: 60.0 }
        "support_channel=chat, priority_support=0": { mean: 12.0, std: 4.0, min: 1.0, max: 50.0 }
        "support_channel=email, priority_support=1": { mean: 22.0, std: 6.0, min: 3.0, max: 70.0 }
        "support_channel=email, priority_support=0": { mean: 15.0, std: 5.0, min: 2.0, max: 60.0 }
        "support_channel=phone, priority_support=1": { mean: 30.0, std: 8.0, min: 5.0, max: 90.0 }
        "support_channel=phone, priority_support=0": { mean: 22.0, std: 6.0, min: 4.0, max: 75.0 }

  - column_id: "risk_score"
    distribution:
      type: "continuous"
      depend_on: ["loyalty", "discount"]
      conditioning_bins:
        edges: [0.0, 20.0, 35.0, 50.0, 65.0, 80.0, 100.0]
        labels: ["r1", "r2", "r3", "r4", "r5", "r6"]
      bin_probs: { r1: 0.10, r2: 0.16, r3: 0.24, r4: 0.24, r5: 0.16, r6: 0.10 }
      conditional_targets:
        "loyalty=1, discount=1": { mean: 35.0, std: 10.0, min: 0.0, max: 100.0 }
        "loyalty=1, discount=0": { mean: 45.0, std: 12.0, min: 0.0, max: 100.0 }
        "loyalty=0, discount=1": { mean: 55.0, std: 15.0, min: 0.0, max: 100.0 }
        "loyalty=0, discount=0": { mean: 65.0, std: 18.0, min: 0.0, max: 100.0 }

  - column_id: "churn_risk"
    values: { true_value: 1, false_value: 0 }
    distribution:
      type: "conditional"
      depend_on: ["plan_tier", "priority_support", "discount"]
      conditional_probs:
        "plan_tier=basic, priority_support=1, discount=1": { true_prob: 0.25, false_prob: 0.75 }
        "plan_tier=basic, priority_support=1, discount=0": { true_prob: 0.35, false_prob: 0.65 }
        "plan_tier=basic, priority_support=0, discount=1": { true_prob: 0.30, false_prob: 0.70 }
        "plan_tier=basic, priority_support=0, discount=0": { true_prob: 0.45, false_prob: 0.55 }
        "plan_tier=plus, priority_support=1, discount=1": { true_prob: 0.18, false_prob: 0.82 }
        "plan_tier=plus, priority_support=1, discount=0": { true_prob: 0.25, false_prob: 0.75 }
        "plan_tier=plus, priority_support=0, discount=1": { true_prob: 0.22, false_prob: 0.78 }
        "plan_tier=plus, priority_support=0, discount=0": { true_prob: 0.32, false_prob: 0.68 }
        "plan_tier=pro, priority_support=1, discount=1": { true_prob: 0.12, false_prob: 0.88 }
        "plan_tier=pro, priority_support=1, discount=0": { true_prob: 0.18, false_prob: 0.82 }
        "plan_tier=pro, priority_support=0, discount=1": { true_prob: 0.15, false_prob: 0.85 }
        "plan_tier=pro, priority_support=0, discount=0": { true_prob: 0.25, false_prob: 0.75 }
"""


_SAMPLE_CONFIGS = {
    "binary": CONFIG_BINARY,
    "categorical": CONFIG_CATEGORICAL,
    "binary_categorical_large": CONFIG_BINARY_CATEGORICAL_LARGE,
    "continuous": CONFIG_CONTINUOUS,
    "mixed": CONFIG_MIXED,
    "continuous_parent_bins": CONFIG_CONTINUOUS_PARENT_BINS,
    "mixed_large": CONFIG_MIXED_LARGE,
}


def available_sample_configs() -> list[str]:
    """Return sorted names for all built-in sample configurations."""

    return sorted(_SAMPLE_CONFIGS.keys())


def load_config(config: Any) -> dict[str, Any]:
    """Parse and normalize a config from YAML text or dict input."""

    if isinstance(config, dict):
        return copy.deepcopy(config)

    if isinstance(config, str):
        parsed = yaml.safe_load(config)
        if parsed is None:
            raise ValueError("Config text is empty")
        if not isinstance(parsed, dict):
            raise ValueError("Config must parse to a mapping")
        return parsed

    raise TypeError("Config must be a dict or YAML string")


def get_sample_config(name: str) -> dict[str, Any]:
    """Load one of the built-in sample configurations by name."""

    key = str(name).strip().lower()
    if key not in _SAMPLE_CONFIGS:
        options = ", ".join(available_sample_configs())
        raise ValueError(f"Unknown sample config '{name}'. Available: {options}")
    return load_config(_SAMPLE_CONFIGS[key])
