import os
import torch
import yaml
import lzma
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from omegaconf import OmegaConf
from hydra.utils import instantiate

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pathlib import Path
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Scene, Trajectory


class PDM_Reward:
    """
    A class that encapsulates the RL PDM reward calculation for gievn token.
    """
    def __init__(self, metric_cache_path):
        """
        Initialize the reward calculator with the given configuration.

        :param metric_cache_path: Path to the metric cache.
        """
        # Initialize the necessary components
        self.metric_cache_loader = MetricCacheLoader(metric_cache_path)
        self.future_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.future_sampling)
        self.scorer= PDMScorer(self.future_sampling)

    def rl_pdm_score(self, trajectory, token, shaped=False, baseline_scores=None,
                     return_components=False, use_true_pdms=False):
        """
        Compute the rl pdm reward for a given token.

        :param trajectory: model output.
        :param token: The scene token.
        :param shaped: If True, return weighted sub-metric delta reward.
                       If False, return raw PDMS product.
        :param baseline_scores: dict of per-scene baseline sub-metric scores for delta computation.
        :param return_components: when True, also return the six raw sub-metrics.
        :param use_true_pdms: If True, return raw PDMS product (result.score) directly.
                              This bypasses the shaped delta reward and uses the exact
                              same PDMScore as navtest eval, eliminating the proxy gap.
        """
        metric_cache_path = self.metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache = pickle.load(f)

        try:
            # Compute the pdm score (same code path as eval)
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=self.future_sampling,
                simulator=self.simulator,
                scorer=self.scorer,
            )

            # v5: use true PDMS product directly as reward
            # This eliminates the simulator-proxy → shaped-delta gap.
            # result.score is the multiplicative PDMS product used in navtest eval.
            if use_true_pdms:
                if return_components:
                    sub_scores = {
                        'collision': result.no_at_fault_collisions,
                        'drivable': result.drivable_area_compliance,
                        'progress': result.ego_progress,
                        'ttc': result.time_to_collision_within_bound,
                        'comfort': result.comfort,
                        'direction': result.driving_direction_compliance,
                    }
                    return result.score, sub_scores
                return result.score

            if not shaped:
                return result.score

            # --- Shaped sub-metric delta reward (legacy) ---
            sub_scores = {
                'collision': result.no_at_fault_collisions,
                'drivable': result.drivable_area_compliance,
                'progress': result.ego_progress,
                'ttc': result.time_to_collision_within_bound,
                'comfort': result.comfort,
                'direction': result.driving_direction_compliance,
            }

            weights = {
                'progress': 0.35,
                'collision': 0.20,
                'drivable': 0.15,
                'ttc': 0.15,
                'direction': 0.10,
                'comfort': 0.05,
            }

            alpha = 0.3
            beta = 0.7

            absolute_reward = sum(weights[k] * sub_scores[k] for k in weights)

            if baseline_scores is not None:
                delta_reward = sum(
                    weights[k] * (sub_scores[k] - baseline_scores.get(k, 0.0))
                    for k in weights
                )
            else:
                delta_reward = 0.0
                alpha = 1.0
                beta = 0.0

            reward = alpha * absolute_reward + beta * delta_reward
            if return_components:
                return reward, sub_scores
            return reward

        except Exception as e:
            print(f"Reward calculation failed: {e}")
            if return_components:
                return 0.0, {}
            return 0.0
