"""Fusion engine — training, evaluation, metrics, and acceptance gates."""

from .evaluator import non_max_suppression, compute_ap, Evaluator
from .trainer import Trainer

__all__ = ['non_max_suppression', 'compute_ap', 'Evaluator', 'Trainer']
