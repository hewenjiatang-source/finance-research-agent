#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/analyze_ablation.py
================================================================================
Ablation experiment analysis script.

Analyzes ablation results along these dimensions:
1. Number of adversarial denoising rounds: 0 vs 1 vs 2 vs 3
2. Self-evolution: off vs on
3. Other modules (optional)

Supports drawing comparison bar and line charts and outputs a statistics report.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # use the Agg backend in environments without a GUI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class AblationAnalyzer:
    """Ablation experiment result analyzer."""

    def __init__(self, output_dir: str = "outputs/evaluation") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    @staticmethod
    def load_results(path: str) -> dict[str, Any]:
        """Load evaluation results from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------------------------------------
    # Adversarial-rounds ablation analysis
    # -----------------------------------------------------------------------
    def analyze_adversarial_rounds(
        self,
        results: dict[str, Any],
        save_prefix: str = "ablation_adversarial",
    ) -> dict[str, Any]:
        """
        Analyze the effect of the number of adversarial denoising rounds on the composite score.

        Args:
            results: evaluation result dict; keys should include "adv_0", "adv_1", "adv_2", "adv_3", etc.
            save_prefix: filename prefix for saved figures.

        Returns:
            Statistical analysis result dict.
        """
        rounds = []
        scores = []

        for key in sorted(results.keys()):
            if key.startswith("adv_"):
                r = int(key.split("_")[1])
                rounds.append(r)
                scores.append(results[key])

        if not rounds:
            print("[AblationAnalyzer] No adversarial-rounds data found")
            return {}

        # Draw the line chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, scores, marker="o", linewidth=2, markersize=8, color="#2E86AB")
        ax.set_xlabel("Adversarial Rounds", fontsize=12)
        ax.set_ylabel("Average Composite Score", fontsize=12)
        ax.set_title("Effect of Adversarial Rounds on Report Quality", fontsize=14)
        ax.set_xticks(rounds)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim([0.0, 1.0])

        save_path = os.path.join(self.output_dir, f"{save_prefix}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[AblationAnalyzer] Chart saved: {save_path}")

        return {
            "dimension": "adversarial_rounds",
            "rounds": rounds,
            "scores": scores,
            "best_round": rounds[np.argmax(scores)],
            "best_score": max(scores),
        }

    # -----------------------------------------------------------------------
    # Self-evolution ablation analysis
    # -----------------------------------------------------------------------
    def analyze_evolution(
        self,
        results: dict[str, Any],
        save_prefix: str = "ablation_evolution",
    ) -> dict[str, Any]:
        """
        Analyze the effect of turning self-evolution on/off.

        Args:
            results: evaluation result dict; should contain the "evo_off" and "evo_on" keys.
            save_prefix: filename prefix for saved figures.

        Returns:
            Statistical analysis result dict.
        """
        labels = []
        scores = []

        for key, label in [("evo_off", "No Evolution"), ("evo_on", "With Evolution")]:
            if key in results:
                labels.append(label)
                scores.append(results[key])

        if not labels:
            print("[AblationAnalyzer] No self-evolution ablation data found")
            return {}

        # Draw the bar chart
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#E94F37", "#6A994E"]
        bars = ax.bar(labels, scores, color=colors, width=0.5, edgecolor="black")
        ax.set_ylabel("Average Composite Score", fontsize=12)
        ax.set_title("Impact of Self-Evolution Engine", fontsize=14)
        ax.set_ylim([0.0, 1.0])

        # Annotate values above the bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(
                f"{score:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        save_path = os.path.join(self.output_dir, f"{save_prefix}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"[AblationAnalyzer] Chart saved: {save_path}")

        return {
            "dimension": "evolution",
            "labels": labels,
            "scores": scores,
            "improvement": scores[1] - scores[0] if len(scores) == 2 else None,
        }

    # -----------------------------------------------------------------------
    # Comprehensive ablation report
    # -----------------------------------------------------------------------
    def generate_report(
        self,
        adversarial_results: dict[str, Any] | None = None,
        evolution_results: dict[str, Any] | None = None,
        output_name: str = "ablation_analysis.json",
    ) -> str:
        """
        Generate a comprehensive ablation analysis report.

        Args:
            adversarial_results: adversarial ablation results.
            evolution_results: evolution ablation results.
            output_name: output JSON file name.

        Returns:
            Path of the saved file.
        """
        report = {
            "adversarial": adversarial_results or {},
            "evolution": evolution_results or {},
        }

        path = os.path.join(self.output_dir, output_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[AblationAnalyzer] Ablation analysis report saved: {path}")
        return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation experiment analysis script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation/analyze_ablation.py --adv_results outputs/evaluation/adv_results.json
        """,
    )
    parser.add_argument(
        "--adv_results",
        type=str,
        default=None,
        help="Path to the adversarial-rounds ablation result JSON (format: {\"adv_0\": 0.6, \"adv_1\": 0.72, ...})",
    )
    parser.add_argument(
        "--evo_results",
        type=str,
        default=None,
        help="Path to the self-evolution ablation result JSON (format: {\"evo_off\": 0.65, \"evo_on\": 0.75})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for charts and reports",
    )
    args = parser.parse_args()

    analyzer = AblationAnalyzer(output_dir=args.output_dir)

    adv_report = None
    evo_report = None

    if args.adv_results and os.path.exists(args.adv_results):
        adv_data = analyzer.load_results(args.adv_results)
        adv_report = analyzer.analyze_adversarial_rounds(adv_data)

    if args.evo_results and os.path.exists(args.evo_results):
        evo_data = analyzer.load_results(args.evo_results)
        evo_report = analyzer.analyze_evolution(evo_data)

    # If no file is provided, generate sample data for a demo
    if adv_report is None and evo_report is None:
        print("[main] No input data provided; generating demo charts from sample data...")
        adv_report = analyzer.analyze_adversarial_rounds(
            {"adv_0": 0.62, "adv_1": 0.71, "adv_2": 0.78, "adv_3": 0.80}
        )
        evo_report = analyzer.analyze_evolution(
            {"evo_off": 0.68, "evo_on": 0.76}
        )

    analyzer.generate_report(adv_report, evo_report)


if __name__ == "__main__":
    main()
