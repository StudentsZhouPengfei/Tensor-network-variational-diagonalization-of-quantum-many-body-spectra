"""Vendored finite-state-automaton AutoMPO implementation."""

from .class_fsa import fsa
from .class_named_data import named_data
from .opr_pool import GenSpinOpr

__all__ = ["GenSpinOpr", "fsa", "named_data"]
