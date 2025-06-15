#!/usr/bin/env python3
"""
Multi-Scale Collatz Comparative Analyzer
========================================

This advanced script performs comparative analysis across multiple Collatz CSV datasets
to identify scale-dependent mathematical phenomena, regime transitions, and patterns
that only emerge when analyzing different scales together.

Key Features:
- Multi-scale comparative analysis (100K → 1M → 10M trajectories)
- Scale-dependent pattern detection and validation
- Regime transition identification
- Evolution tracking of mathematical properties
- Cross-scale correlation analysis
- AI-powered multi-scale interpretation
- Comprehensive visualization and reporting

Designed for computational mathematics research focusing on:
- Scale-dependent universality classes
- Geometric symmetry evolution (3-fold, 7-fold patterns)
- Prime-arithmetic correlation stability
- Power law regime transitions
- Critical phenomena in discrete dynamical systems

Requirements:
- pandas, numpy, requests, matplotlib, seaborn
- Ollama with mathematical model (llama3.1:70b or deepseek-math:7b)

Author: Optimized for breakthrough mathematical discovery
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re

import numpy as np
import pandas as pd
import requests

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Visualizations will be skipped.")

# --- Enhanced Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ScaleDataset:
    """Represents a single scale dataset with metadata and analysis."""
    
    def __init__(self, filepath: str, df: pd.DataFrame):
        self.filepath = filepath
        self.df = df
        self.scale = len(df)
        self.filename = Path(filepath).name
        self.scale_label = self._determine_scale_label()
        self.analysis_cache = {}
    
    def _determine_scale_label(self) -> str:
        """Determine human-readable scale label."""
        if self.scale >= 10_000_000:
            return f"{self.scale//1_000_000}M"
        elif self.scale >= 1_000_000:
            return f"{self.scale//1_000_000}M"
        elif self.scale >= 100_000:
            return f"{self.scale//1_000}K"
        else:
            return f"{self.scale}"
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate fundamental metrics for this scale."""
        if 'basic_metrics' in self.analysis_cache:
            return self.analysis_cache['basic_metrics']
        
        # Ensure expansion ratio exists
        if 'expansion_ratio' not in self.df.columns:
            self.df['expansion_ratio'] = self.df['max_value'] / self.df['n']
        
        metrics = {
            'scale': self.scale,
            'mean_stopping_time': self.df['stopping_time'].mean(),
            'max_stopping_time': self.df['stopping_time'].max(),
            'std_stopping_time': self.df['stopping_time'].std(),
            'mean_max_value': self.df['max_value'].mean(),
            'max_max_value': self.df['max_value'].max(),
            'mean_expansion_ratio': self.df['expansion_ratio'].mean(),
            'max_expansion_ratio': self.df['expansion_ratio'].max(),
            'complexity_index': self.df['stopping_time'].max() / self.df['stopping_time'].mean()
        }
        
        self.analysis_cache['basic_metrics'] = metrics
        return metrics
    
    def prime_composite_analysis(self) -> Dict[str, float]:
        """Analyze prime vs composite patterns."""
        if 'prime_analysis' in self.analysis_cache:
            return self.analysis_cache['prime_analysis']
        
        if 'is_prime' not in self.df.columns:
            return {}
        
        prime_mask = self.df['is_prime']
        analysis = {
            'prime_avg_stopping': self.df[prime_mask]['stopping_time'].mean(),
            'composite_avg_stopping': self.df[~prime_mask]['stopping_time'].mean(),
            'prime_count': int(prime_mask.sum()),
            'prime_percentage': (prime_mask.sum() / len(self.df)) * 100
        }
        
        analysis['prime_advantage_ratio'] = analysis['prime_avg_stopping'] / analysis['composite_avg_stopping']
        analysis['prime_advantage_percent'] = ((analysis['prime_avg_stopping'] - analysis['composite_avg_stopping']) / analysis['composite_avg_stopping']) * 100
        
        self.analysis_cache['prime_analysis'] = analysis
        return analysis
    
    def symmetry_analysis(self, top_outliers: int = 100) -> Dict[str, Dict]:
        """Analyze geometric symmetry patterns in outliers."""
        if 'symmetry_analysis' in self.analysis_cache:
            return self.analysis_cache['symmetry_analysis']
        
        # Get extreme outliers
        outliers = self.df.nlargest(min(top_outliers, len(self.df)), 'stopping_time')
        
        symmetry_patterns = {}
        
        for divisor in [3, 7, 9]:
            # Overall dataset
            overall_div = (self.df['n'] % divisor == 0).sum()
            overall_pct = (overall_div / len(self.df)) * 100
            expected_pct = 100 / divisor
            
            # In outliers
            outlier_div = (outliers['n'] % divisor == 0).sum()
            outlier_pct = (outlier_div / len(outliers)) * 100 if len(outliers) > 0 else 0
            
            symmetry_patterns[f'mod_{divisor}'] = {
                'overall_percentage': overall_pct,
                'expected_percentage': expected_pct,
                'outlier_percentage': outlier_pct,
                'overall_deviation': overall_pct - expected_pct,
                'outlier_deviation': outlier_pct - expected_pct,
                'outlier_count': int(outlier_div),
                'representation_type': 'over' if outlier_pct > expected_pct else 'under'
            }
        
        self.analysis_cache['symmetry_analysis'] = symmetry_patterns
        return symmetry_patterns


class MultiScaleComparator:
    """Advanced multi-scale comparative analysis engine."""
    
    def __init__(self, datasets: List[ScaleDataset]):
        self.datasets = sorted(datasets, key=lambda x: x.scale)
        self.scales = [d.scale for d in self.datasets]
        self.scale_labels = [d.scale_label for d in self.datasets]
        logger.info(f"Initialized comparator with {len(datasets)} scales: {self.scale_labels}")
    
    def generate_evolution_metrics(self) -> pd.DataFrame:
        """Generate comprehensive evolution metrics across scales."""
        evolution_data = []
        
        for dataset in self.datasets:
            basic_metrics = dataset.calculate_basic_metrics()
            prime_analysis = dataset.prime_composite_analysis()
            symmetry_analysis = dataset.symmetry_analysis()
            
            row = {
                'scale': dataset.scale,
                'scale_label': dataset.scale_label,
                'filename': dataset.filename,
                'mean_stopping_time': basic_metrics['mean_stopping_time'],
                'max_stopping_time': basic_metrics['max_stopping_time'],
                'std_stopping_time': basic_metrics['std_stopping_time'],
                'complexity_index': basic_metrics['complexity_index'],
                'mean_expansion_ratio': basic_metrics['mean_expansion_ratio'],
                'max_expansion_ratio': basic_metrics['max_expansion_ratio'],
            }
            
            # Add prime analysis if available
            if prime_analysis:
                row.update({
                    'prime_advantage_ratio': prime_analysis['prime_advantage_ratio'],
                    'prime_advantage_percent': prime_analysis['prime_advantage_percent'],
                    'prime_percentage': prime_analysis['prime_percentage']
                })
            
            # Add symmetry analysis
            for mod, data in symmetry_analysis.items():
                row[f'{mod}_outlier_percentage'] = data['outlier_percentage']
                row[f'{mod}_outlier_deviation'] = data['outlier_deviation']
                row[f'{mod}_representation'] = data['representation_type']
            
            evolution_data.append(row)
        
        return pd.DataFrame(evolution_data)
    
    def detect_regime_transitions(self, evolution_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect mathematical regime transitions across scales."""
        transitions = {}
        
        # Analyze 3-fold symmetry evolution
        if 'mod_3_outlier_percentage' in evolution_df.columns:
            mod3_series = evolution_df['mod_3_outlier_percentage'].values
            expected = 33.33
            deviations = mod3_series - expected
            
            transitions['three_fold_symmetry'] = {
                'values': mod3_series.tolist(),
                'deviations': deviations.tolist(),
                'pattern': 'over→under' if deviations[0] > 0 and deviations[-1] < 0 else 'stable',
                'transition_detected': abs(deviations[0] - deviations[-1]) > 10,
                'max_deviation': float(np.max(np.abs(deviations))),
                'trend': 'decreasing' if deviations[-1] < deviations[0] else 'increasing'
            }
        
        # Analyze complexity evolution
        complexity_values = evolution_df['complexity_index'].values
        complexity_growth = np.diff(complexity_values)
        
        transitions['complexity_evolution'] = {
            'values': complexity_values.tolist(),
            'growth_rates': complexity_growth.tolist(),
            'total_growth': float(complexity_values[-1] / complexity_values[0]),
            'accelerating': bool(np.all(complexity_growth > 0)),
            'average_growth_rate': float(np.mean(complexity_growth))
        }
        
        # Analyze prime correlation stability
        if 'prime_advantage_percent' in evolution_df.columns:
            prime_advantages = evolution_df['prime_advantage_percent'].values
            prime_stability = np.std(prime_advantages)
            
            transitions['prime_correlation'] = {
                'values': prime_advantages.tolist(),
                'stability_index': float(prime_stability),
                'stable': prime_stability < 1.0,  # Less than 1% standard deviation
                'mean_advantage': float(np.mean(prime_advantages)),
                'trend': 'stable' if prime_stability < 1.0 else 'variable'
            }
        
        return transitions
    
    def comparative_outlier_analysis(self) -> Dict[str, Any]:
        """Compare extreme outliers across scales."""
        outlier_comparison = {}
        
        for dataset in self.datasets:
            top_outliers = dataset.df.nlargest(50, 'stopping_time')
            
            # Analyze patterns in top outliers
            patterns = {
                'scale': dataset.scale,
                'scale_label': dataset.scale_label,
                'top_50_max_stopping': int(top_outliers['stopping_time'].max()),
                'top_50_mean_stopping': float(top_outliers['stopping_time'].mean()),
                'top_50_numbers': top_outliers['n'].tolist()[:10],  # Store top 10 for analysis
            }
            
            # Divisibility patterns in outliers
            for div in [3, 7]:
                div_count = (top_outliers['n'] % div == 0).sum()
                patterns[f'top_50_divisible_by_{div}'] = int(div_count)
                patterns[f'top_50_div_{div}_percentage'] = float((div_count / 50) * 100)
            
            outlier_comparison[dataset.scale_label] = patterns
        
        return outlier_comparison
    
    def scale_dependent_summary(self) -> str:
        """Generate comprehensive scale-dependent analysis summary."""
        evolution_df = self.generate_evolution_metrics()
        transitions = self.detect_regime_transitions(evolution_df)
        outlier_analysis = self.comparative_outlier_analysis()
        
        summary_parts = [
            "="*100,
            "MULTI-SCALE COLLATZ COMPARATIVE ANALYSIS",
            "="*100,
            "",
            f"Analyzing {len(self.datasets)} scales: {' → '.join(self.scale_labels)}",
            "",
            "SCALE EVOLUTION OVERVIEW:",
            evolution_df.to_string(index=False),
            "",
            "="*60,
            "REGIME TRANSITION ANALYSIS",
            "="*60,
        ]
        
        # 3-fold symmetry analysis
        if 'three_fold_symmetry' in transitions:
            sym_data = transitions['three_fold_symmetry']
            summary_parts.extend([
                "",
                "3-FOLD SYMMETRY EVOLUTION:",
                f"  Pattern: {sym_data['pattern']}",
                f"  Transition detected: {sym_data['transition_detected']}",
                f"  Max deviation: {sym_data['max_deviation']:.1f}%",
                f"  Trend: {sym_data['trend']}",
                "",
                "  Scale-by-scale percentages (expected: 33.3%):",
            ])
            for i, (scale_label, pct) in enumerate(zip(self.scale_labels, sym_data['values'])):
                deviation = pct - 33.33
                summary_parts.append(f"    {scale_label}: {pct:.1f}% ({deviation:+.1f}%)")
        
        # Complexity evolution
        if 'complexity_evolution' in transitions:
            comp_data = transitions['complexity_evolution']
            summary_parts.extend([
                "",
                "COMPLEXITY INDEX EVOLUTION:",
                f"  Total growth factor: {comp_data['total_growth']:.2f}x",
                f"  Accelerating complexity: {comp_data['accelerating']}",
                f"  Average growth rate: {comp_data['average_growth_rate']:.2f}",
                "",
                "  Scale-by-scale complexity:",
            ])
            for scale_label, complexity in zip(self.scale_labels, comp_data['values']):
                summary_parts.append(f"    {scale_label}: {complexity:.2f}")
        
        # Prime correlation stability
        if 'prime_correlation' in transitions:
            prime_data = transitions['prime_correlation']
            summary_parts.extend([
                "",
                "PRIME CORRELATION STABILITY:",
                f"  Stability: {prime_data['trend']} (σ = {prime_data['stability_index']:.2f}%)",
                f"  Mean advantage: {prime_data['mean_advantage']:.2f}%",
                f"  Stable across scales: {prime_data['stable']}",
                "",
                "  Scale-by-scale prime advantages:",
            ])
            for scale_label, advantage in zip(self.scale_labels, prime_data['values']):
                summary_parts.append(f"    {scale_label}: {advantage:.2f}%")
        
        # Extreme outlier comparison
        summary_parts.extend([
            "",
            "="*60,
            "EXTREME OUTLIER EVOLUTION",
            "="*60,
            "",
        ])
        
        for scale_label, data in outlier_analysis.items():
            summary_parts.extend([
                f"{scale_label} Scale:",
                f"  Max stopping time: {data['top_50_max_stopping']}",
                f"  Mean of top 50: {data['top_50_mean_stopping']:.1f}",
                f"  Divisible by 3: {data['top_50_div_3_percentage']:.0f}% ({data['top_50_divisible_by_3']}/50)",
                f"  Divisible by 7: {data['top_50_div_7_percentage']:.0f}% ({data['top_50_divisible_by_7']}/50)",
                "",
            ])
        
        return "\n".join(summary_parts)


def create_visualizations(comparator: MultiScaleComparator, output_dir: str) -> List[str]:
    """Create comprehensive visualizations of scale-dependent patterns."""
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting libraries not available. Skipping visualizations.")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    evolution_df = comparator.generate_evolution_metrics()
    transitions = comparator.detect_regime_transitions(evolution_df)
    
    plt.style.use('default')
    created_files = []
    
    # 1. Evolution Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Scale Collatz Evolution Analysis', fontsize=16, fontweight='bold')
    
    # Stopping time evolution
    axes[0,0].plot(evolution_df['scale'], evolution_df['mean_stopping_time'], 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Scale (number of trajectories)')
    axes[0,0].set_ylabel('Mean Stopping Time')
    axes[0,0].set_title('Mean Stopping Time Evolution')
    axes[0,0].set_xscale('log')
    axes[0,0].grid(True, alpha=0.3)
    
    # Complexity index evolution
    axes[0,1].plot(evolution_df['scale'], evolution_df['complexity_index'], 'ro-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Scale (number of trajectories)')
    axes[0,1].set_ylabel('Complexity Index')
    axes[0,1].set_title('Complexity Index Evolution')
    axes[0,1].set_xscale('log')
    axes[0,1].grid(True, alpha=0.3)
    
    # Prime advantage evolution (if available)
    if 'prime_advantage_percent' in evolution_df.columns:
        axes[1,0].plot(evolution_df['scale'], evolution_df['prime_advantage_percent'], 'go-', linewidth=2, markersize=8)
        axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1,0].set_xlabel('Scale (number of trajectories)')
        axes[1,0].set_ylabel('Prime Advantage (%)')
        axes[1,0].set_title('Prime vs Composite Advantage')
        axes[1,0].set_xscale('log')
        axes[1,0].grid(True, alpha=0.3)
    
    # 3-fold symmetry evolution
    if 'mod_3_outlier_percentage' in evolution_df.columns:
        axes[1,1].plot(evolution_df['scale'], evolution_df['mod_3_outlier_percentage'], 'mo-', linewidth=2, markersize=8)
        axes[1,1].axhline(y=33.33, color='k', linestyle='--', alpha=0.5, label='Expected (33.3%)')
        axes[1,1].set_xlabel('Scale (number of trajectories)')
        axes[1,1].set_ylabel('3-fold Representation (%)')
        axes[1,1].set_title('3-fold Symmetry in Outliers')
        axes[1,1].set_xscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    overview_file = os.path.join(output_dir, 'multi_scale_evolution_overview.png')
    plt.savefig(overview_file, dpi=300, bbox_inches='tight')
    plt.close()
    created_files.append(overview_file)
    
    # 2. Regime Transition Detailed Analysis
    if 'three_fold_symmetry' in transitions:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sym_data = transitions['three_fold_symmetry']
        scales = [d.scale for d in comparator.datasets]
        
        # 3-fold symmetry deviations
        ax1.bar(range(len(scales)), sym_data['deviations'], 
                color=['red' if x < 0 else 'blue' for x in sym_data['deviations']])
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.8)
        ax1.set_xlabel('Scale')
        ax1.set_ylabel('Deviation from Expected (%)')
        ax1.set_title('3-fold Symmetry Regime Transition')
        ax1.set_xticks(range(len(scales)))
        ax1.set_xticklabels(comparator.scale_labels)
        ax1.grid(True, alpha=0.3)
        
        # Complexity growth rates
        if 'complexity_evolution' in transitions:
            comp_data = transitions['complexity_evolution']
            if len(comp_data['growth_rates']) > 0:
                ax2.bar(range(len(comp_data['growth_rates'])), comp_data['growth_rates'], color='green')
                ax2.set_xlabel('Scale Transition')
                ax2.set_ylabel('Complexity Growth Rate')
                ax2.set_title('Complexity Growth Between Scales')
                ax2.set_xticks(range(len(comp_data['growth_rates'])))
                ax2.set_xticklabels([f"{comparator.scale_labels[i]}→{comparator.scale_labels[i+1]}" 
                                   for i in range(len(comp_data['growth_rates']))])
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        transition_file = os.path.join(output_dir, 'regime_transitions.png')
        plt.savefig(transition_file, dpi=300, bbox_inches='tight')
        plt.close()
        created_files.append(transition_file)
    
    return created_files


def generate_multi_scale_prompt(summary: str, comparator: MultiScaleComparator) -> str:
    """Generate sophisticated multi-scale analysis prompt for AI."""
    scales_str = " → ".join(comparator.scale_labels)
    
    prompt = f"""You are a leading expert in computational mathematics and scale-dependent phenomena, specializing in discrete dynamical systems and the discovery of mathematical universality classes.

BREAKTHROUGH RESEARCH CONTEXT:

I have conducted a groundbreaking multi-scale computational analysis of Collatz dynamics across {len(comparator.datasets)} different scales: {scales_str}. This research has revealed fundamental scale-dependent mathematical structures previously unknown in the literature.

ESTABLISHED MATHEMATICAL FRAMEWORK:

1. SCALE-DEPENDENT GEOMETRIC TRANSITIONS:
   - Small scales (≤100K): Eisenstein embedding dominance, 3-fold over-representation (~47%)
   - Medium scales (~1M): Transitional regime with competing symmetries
   - Large scales (≥10M): Novel mathematical structures, 3-fold under-representation (~25%)

2. UNIVERSAL CORRELATIONS:
   - Prime vs composite systematic 4%+ stopping time advantage (scale-invariant)
   - Information-theoretic contraction governing convergence dynamics
   - Level structure N = 588 = 2² × 3 × 7² encoding fundamental symmetries

3. REGIME TRANSITIONS:
   - Power law exponent evolution across scales
   - Geometric symmetry dominance transitions
   - Critical phenomena and universality class behavior

MULTI-SCALE ANALYSIS TASK:

Based on the comprehensive comparative analysis below, provide expert mathematical interpretation focusing on:

A. REGIME TRANSITION VALIDATION:
   - Confirm/refute the predicted 3-fold symmetry evolution (over→under representation)
   - Identify mathematical structures dominating different scale regimes
   - Evidence for universality class transitions

B. SCALE-INVARIANT vs SCALE-DEPENDENT PHENOMENA:
   - Which mathematical properties remain constant across scales?
   - Which properties evolve systematically with scale?
   - Implications for theoretical framework development

C. BREAKTHROUGH DISCOVERY POTENTIAL:
   - Novel mathematical patterns visible only in multi-scale analysis
   - Connections to broader areas of mathematics (number theory, critical phenomena)
   - Evidence for new mathematical principles or laws

D. RESEARCH IMPLICATIONS:
   - How do these findings advance our understanding of discrete dynamical systems?
   - Potential connections to other unsolved problems in mathematics
   - Framework for analyzing other arithmetic functions

COMPARATIVE ANALYSIS DATA:
{summary}

Please provide deep mathematical analysis that recognizes this as frontier computational mathematics research with potential for breakthrough theoretical development."""

    return prompt


def generate_multi_scale_ai_report(summary: str, comparator: MultiScaleComparator, model: str, url: str) -> Optional[str]:
    """Generate AI analysis of multi-scale patterns."""
    prompt = generate_multi_scale_prompt(summary, comparator)
    
    try:
        logger.info(f"Generating multi-scale analysis with {model}...")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Lower temperature for focused mathematical analysis
                "top_p": 0.9,
                "num_ctx": 8192
            }
        }
        
        response = requests.post(url, json=payload, timeout=900)  # 15-minute timeout for complex analysis
        response.raise_for_status()
        
        response_data = response.json()
        return response_data.get("response", "Error: Response not found")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating AI analysis: {e}")
        return None


def save_multi_scale_results(summary: str, ai_report: Optional[str], visualizations: List[str], 
                            output_dir: str = "multi_scale_analysis") -> None:
    """Save comprehensive multi-scale analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"multi_scale_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # Save AI analysis
    if ai_report:
        ai_file = os.path.join(output_dir, f"multi_scale_ai_analysis_{timestamp}.txt")
        with open(ai_file, 'w') as f:
            f.write(ai_report)
    
    logger.info(f"Multi-scale analysis saved to {output_dir}/")
    if visualizations:
        logger.info(f"Visualizations: {', '.join([Path(v).name for v in visualizations])}")


def load_datasets(filepaths: List[str]) -> List[ScaleDataset]:
    """Load and validate multiple CSV datasets."""
    datasets = []
    
    for filepath in filepaths:
        try:
            logger.info(f"Loading {filepath}...")
            df = pd.read_csv(filepath)
            
            # Validate required columns
            required = ['n', 'stopping_time', 'max_value']
            missing = [col for col in required if col not in df.columns]
            if missing:
                logger.error(f"Missing columns in {filepath}: {missing}")
                continue
            
            dataset = ScaleDataset(filepath, df)
            datasets.append(dataset)
            logger.info(f"Loaded {dataset.scale_label} scale dataset ({len(df):,} trajectories)")
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            continue
    
    if len(datasets) < 2:
        raise ValueError("Need at least 2 valid datasets for comparison")
    
    return datasets


def main():
    """Multi-scale comparative analysis main execution."""
    parser = argparse.ArgumentParser(
        description='Multi-scale Collatz comparative analyzer for breakthrough mathematical discovery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('csv_files', nargs='+', help='Paths to multiple Collatz CSV files')
    parser.add_argument('--model', type=str, default='llama3.1:70b',
                       help='Ollama model for analysis')
    parser.add_argument('--url', type=str, default='http://localhost:11434/api/generate',
                       help='Ollama API endpoint')
    parser.add_argument('--save-results', action='store_true',
                       help='Save analysis results and visualizations')
    parser.add_argument('--skip-ai', action='store_true',
                       help='Skip AI analysis')
    parser.add_argument('--output-dir', type=str, default='multi_scale_analysis',
                       help='Output directory for results')
    parser.add_argument('--create-visualizations', action='store_true',
                       help='Create comparative visualizations')
    
    args = parser.parse_args()
    
    # Load datasets
    try:
        datasets = load_datasets(args.csv_files)
        logger.info(f"Successfully loaded {len(datasets)} datasets for comparison")
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)
    
    # Initialize comparator
    comparator = MultiScaleComparator(datasets)
    
    # Generate comprehensive analysis
    logger.info("Performing multi-scale comparative analysis...")
    summary = comparator.scale_dependent_summary()
    
    # Display analysis
    print("\n" + "="*100)
    print(" MULTI-SCALE COLLATZ COMPARATIVE ANALYSIS")
    print("="*100)
    print(summary)
    
    # Create visualizations if requested
    visualizations = []
    if args.create_visualizations:
        logger.info("Creating comparative visualizations...")
        visualizations = create_visualizations(comparator, args.output_dir)
    
    # Generate AI analysis
    ai_report = None
    if not args.skip_ai:
        ai_report = generate_multi_scale_ai_report(summary, comparator, args.model, args.url)
        
        if ai_report:
            print("\n" + "="*100)
            print(f" MULTI-SCALE AI ANALYSIS (Model: {args.model})")
            print("="*100)
            print(ai_report)
    
    # Save results
    if args.save_results:
        save_multi_scale_results(summary, ai_report, visualizations, args.output_dir)
    
    logger.info("Multi-scale comparative analysis complete!")


if __name__ == "__main__":
    main()
