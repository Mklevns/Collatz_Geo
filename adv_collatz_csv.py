#!/usr/bin/env python3
"""
Advanced Collatz CSV Data Analyzer with Mathematical Intelligence
================================================================

This enhanced script provides comprehensive analysis of Collatz trajectory data
with specialized mathematical pattern detection, scale-dependent analysis,
and intelligent AI collaboration for research-grade insights.

Key Features:
- Adaptive outlier analysis based on dataset size
- Mathematical pattern detection (primes, divisibility, residue classes)
- Scale-dependent comparison and reporting
- Intelligent prompt engineering for mathematical AI collaboration
- Research-quality output generation and saving
- Comprehensive statistical validation

Requirements:
- pandas: pip install pandas
- requests: pip install requests
- numpy: pip install numpy
- A running Ollama instance with mathematical model (recommended: llama3.1:70b or deepseek-math:7b)

Author: Optimized for computational mathematics research
"""

import argparse
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# --- Setup Enhanced Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MathematicalAnalyzer:
    """Advanced mathematical analysis for Collatz trajectory data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.total_trajectories = len(df)
        self.validate_data()
    
    def validate_data(self) -> None:
        """Validate the DataFrame has required columns."""
        required_cols = ['n', 'stopping_time', 'max_value']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Dataset validated: {self.total_trajectories:,} trajectories")
    
    def calculate_expansion_ratio(self) -> None:
        """Calculate expansion ratio if not present."""
        if 'expansion_ratio' not in self.df.columns:
            self.df['expansion_ratio'] = self.df['max_value'] / self.df['n']
            logger.info("Calculated expansion_ratio column")
    
    def prime_composite_analysis(self) -> Dict[str, float]:
        """Analyze prime vs composite stopping time differences."""
        if 'is_prime' not in self.df.columns:
            logger.warning("No 'is_prime' column found, skipping prime analysis")
            return {}
        
        prime_mask = self.df['is_prime']
        prime_avg = self.df[prime_mask]['stopping_time'].mean()
        composite_avg = self.df[~prime_mask]['stopping_time'].mean()
        prime_count = prime_mask.sum()
        
        return {
            'prime_avg_stopping_time': prime_avg,
            'composite_avg_stopping_time': composite_avg,
            'prime_advantage_ratio': prime_avg / composite_avg,
            'prime_advantage_percent': ((prime_avg - composite_avg) / composite_avg) * 100,
            'prime_count': int(prime_count),
            'prime_percentage': (prime_count / self.total_trajectories) * 100
        }
    
    def divisibility_analysis(self) -> Dict[str, Dict]:
        """Analyze divisibility patterns in extreme outliers."""
        patterns = {}
        
        # Get top outliers for analysis
        top_stopping = self.df.nlargest(100, 'stopping_time')
        top_expansion = self.df.nlargest(100, 'expansion_ratio')
        
        for divisor in [3, 7, 9]:
            # Overall dataset
            overall_div = (self.df['n'] % divisor == 0).sum()
            overall_pct = (overall_div / self.total_trajectories) * 100
            expected_pct = 100 / divisor
            
            # In extreme outliers
            stopping_div = (top_stopping['n'] % divisor == 0).sum()
            expansion_div = (top_expansion['n'] % divisor == 0).sum()
            
            patterns[f'divisible_by_{divisor}'] = {
                'overall_percentage': overall_pct,
                'expected_percentage': expected_pct,
                'over_under_representation': overall_pct - expected_pct,
                'in_top_stopping_outliers': stopping_div,
                'in_top_expansion_outliers': expansion_div,
                'outlier_representation': (stopping_div / 100) * 100,
                'significance': 'over' if overall_pct > expected_pct else 'under'
            }
        
        return patterns
    
    def scale_comparison_analysis(self) -> Dict[str, float]:
        """Provide context for scale-dependent analysis."""
        mean_stopping = self.df['stopping_time'].mean()
        max_stopping = self.df['stopping_time'].max()
        std_stopping = self.df['stopping_time'].std()
        
        # Historical context (you can update these with your actual measurements)
        scale_context = {
            'current_scale': self.total_trajectories,
            'current_mean_stopping': mean_stopping,
            'current_max_stopping': max_stopping,
            'current_std_stopping': std_stopping,
            'complexity_index': max_stopping / mean_stopping,  # Measure of extreme behavior
        }
        
        # Add historical comparison if we know the scale
        if self.total_trajectories >= 1000000:
            scale_context['scale_regime'] = 'Large Scale (1M+)'
            scale_context['expected_properties'] = 'Multi-regime transitions, 3-fold under-representation in extremes'
        elif self.total_trajectories >= 100000:
            scale_context['scale_regime'] = 'Medium Scale (100K+)'
            scale_context['expected_properties'] = 'Transitional behavior, evolving power laws'
        else:
            scale_context['scale_regime'] = 'Small Scale (<100K)'
            scale_context['expected_properties'] = 'Eisenstein dominance, 3-fold over-representation'
        
        return scale_context
    
    def detect_mathematical_patterns(self) -> Dict[str, any]:
        """Detect specific mathematical patterns in outliers."""
        patterns = {}
        
        # Get extreme outliers
        top_outliers = self.df.nlargest(50, 'stopping_time')
        
        # Analyze prime factorization patterns
        small_prime_factors = []
        for n in top_outliers['n'].head(20):  # Check top 20 for computational efficiency
            factors = []
            temp_n = n
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
                while temp_n % p == 0:
                    factors.append(p)
                    temp_n //= p
                if temp_n == 1:
                    break
            small_prime_factors.append(max(factors) if factors else n)
        
        patterns['largest_small_prime_factors'] = small_prime_factors
        patterns['avg_largest_small_prime'] = np.mean(small_prime_factors) if small_prime_factors else 0
        
        # Residue class analysis
        for mod in [3, 7, 9]:
            residues = top_outliers['n'] % mod
            patterns[f'residue_classes_mod_{mod}'] = residues.value_counts().to_dict()
        
        return patterns


def intelligent_outlier_selection(df: pd.DataFrame) -> int:
    """Intelligently select number of outliers based on dataset size."""
    size = len(df)
    if size >= 10000000:
        return 1000  # For 10M+ datasets
    elif size >= 1000000:
        return 500   # For 1M+ datasets
    elif size >= 100000:
        return 200   # For 100K+ datasets
    else:
        return min(100, size // 10)  # For smaller datasets


def generate_comprehensive_summary(analyzer: MathematicalAnalyzer, top_n: int) -> str:
    """Generate comprehensive mathematical summary."""
    df = analyzer.df
    
    # Calculate expansion ratio if needed
    analyzer.calculate_expansion_ratio()
    
    # Basic statistics
    basic_stats = df[['stopping_time', 'max_value', 'expansion_ratio']].describe()
    
    # Mathematical analyses
    prime_analysis = analyzer.prime_composite_analysis()
    divisibility_patterns = analyzer.divisibility_analysis()
    scale_analysis = analyzer.scale_comparison_analysis()
    mathematical_patterns = analyzer.detect_mathematical_patterns()
    
    # Get outliers
    top_stopping = df.nlargest(min(top_n, 100), 'stopping_time')[['n', 'stopping_time']]
    top_max_value = df.nlargest(min(top_n, 20), 'max_value')[['n', 'max_value', 'stopping_time']]
    top_expansion = df.nlargest(min(top_n, 20), 'expansion_ratio')[['n', 'expansion_ratio', 'stopping_time']]
    
    # Format summary
    summary_parts = [
        "="*80,
        "COMPREHENSIVE COLLATZ MATHEMATICAL ANALYSIS",
        "="*80,
        "",
        f"Dataset: {len(df):,} trajectories",
        f"Scale Regime: {scale_analysis.get('scale_regime', 'Unknown')}",
        f"Analysis Properties: {scale_analysis.get('expected_properties', 'Standard analysis')}",
        "",
        "BASIC STATISTICAL DISTRIBUTION:",
        basic_stats.to_string(),
        "",
        "SCALE-DEPENDENT METRICS:",
        f"  Mean stopping time: {scale_analysis['current_mean_stopping']:.2f}",
        f"  Maximum stopping time: {scale_analysis['current_max_stopping']}",
        f"  Complexity index: {scale_analysis['complexity_index']:.2f}",
        "",
    ]
    
    # Prime analysis section
    if prime_analysis:
        summary_parts.extend([
            "PRIME vs COMPOSITE ANALYSIS:",
            f"  Prime average stopping time: {prime_analysis['prime_avg_stopping_time']:.2f}",
            f"  Composite average stopping time: {prime_analysis['composite_avg_stopping_time']:.2f}",
            f"  Prime advantage ratio: {prime_analysis['prime_advantage_ratio']:.4f}",
            f"  Prime advantage: {prime_analysis['prime_advantage_percent']:.2f}%",
            f"  Primes in dataset: {prime_analysis['prime_count']:,} ({prime_analysis['prime_percentage']:.1f}%)",
            "",
        ])
    
    # Divisibility analysis
    summary_parts.extend([
        "GEOMETRIC SYMMETRY ANALYSIS (Divisibility Patterns):",
    ])
    
    for div, pattern in divisibility_patterns.items():
        div_num = div.split('_')[-1]
        summary_parts.extend([
            f"  Divisible by {div_num}:",
            f"    Overall: {pattern['overall_percentage']:.1f}% (expected: {pattern['expected_percentage']:.1f}%)",
            f"    {pattern['significance'].upper()}-representation by {abs(pattern['over_under_representation']):.1f}%",
            f"    In top outliers: {pattern['outlier_representation']:.0f}%",
        ])
    
    summary_parts.extend([
        "",
        "EXTREME OUTLIER ANALYSIS:",
        "",
        "Top Numbers by Stopping Time:",
        top_stopping.to_string(index=False),
        "",
        "Top Numbers by Peak Value:",
        top_max_value.to_string(index=False),
        "",
        "Top Numbers by Expansion Ratio:",
        top_expansion.to_string(index=False),
        "",
    ])
    
    # Mathematical patterns
    if mathematical_patterns.get('largest_small_prime_factors'):
        summary_parts.extend([
            "MATHEMATICAL PATTERN ANALYSIS:",
            f"  Average largest small prime factor in extremes: {mathematical_patterns['avg_largest_small_prime']:.1f}",
            "",
        ])
    
    return "\n".join(summary_parts)


def generate_enhanced_prompt(summary: str, analyzer: MathematicalAnalyzer) -> str:
    """Generate mathematically sophisticated prompt for AI analysis."""
    total = analyzer.total_trajectories
    
    prompt = f"""You are a specialist in computational number theory and discrete dynamical systems, with expertise in the Collatz conjecture and scale-dependent mathematical phenomena.

RESEARCH CONTEXT:
I have analyzed {total:,} Collatz trajectories as part of a multi-scale computational mathematics study. This research has discovered:

1. SCALE-DEPENDENT GEOMETRIC TRANSITIONS: Different mathematical structures dominate at different scales
   - Small scale (≤100K): 3-fold symmetry over-representation (~47% vs 33% expected)
   - Medium scale (~1M): Transitional behavior with evolving power laws  
   - Large scale (≥10M): 3-fold under-representation (~25% vs 33% expected)

2. PRIME-ARITHMETIC CORRELATIONS: Systematic 4%+ advantage in prime stopping times
   - This represents a genuine number-theoretic connection
   - Validates bridges between discrete dynamics and prime theory

3. GEOMETRIC FRAMEWORK: Eisenstein plane embedding reveals crystalline structure
   - Level structure N = 588 = 2² × 3 × 7² encodes fundamental symmetries
   - Each factor (2², 3, 7²) governs different aspects of the dynamics

ANALYSIS TASK:
Based on the statistical summary below, provide research-grade mathematical analysis focusing on:

A. MATHEMATICAL PATTERN RECOGNITION:
   - Divisibility patterns in extreme outliers (especially multiples of 3, 7)
   - Prime factorization characteristics of extreme cases
   - Residue class distributions and their significance

B. SCALE-DEPENDENT PHENOMENA:
   - How do these results compare to the expected scale-dependent behavior?
   - What mathematical structures appear to be dominating at this scale?
   - Evidence for regime transitions or universality classes

C. NUMBER-THEORETIC CONNECTIONS:
   - Prime vs composite systematic differences
   - Multiplicative structure patterns in outliers
   - Connections to classical number theory (modular arithmetic, L-functions)

D. RESEARCH IMPLICATIONS:
   - What new mathematical questions do these patterns suggest?
   - How might these findings connect to broader areas of mathematics?
   - Potential for breakthrough theoretical development

STATISTICAL SUMMARY:
{summary}

Please provide deep mathematical analysis that recognizes the sophistication of this computational mathematics research."""

    return prompt


def generate_ollama_report(summary: str, analyzer: MathematicalAnalyzer, model: str, url: str) -> Optional[str]:
    """Generate AI report using enhanced mathematical prompt."""
    prompt = generate_enhanced_prompt(summary, analyzer)
    
    try:
        logger.info(f"Generating research-grade analysis with {model}...")
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for more focused mathematical analysis
                "top_p": 0.9,
                "num_ctx": 8192  # Larger context window for comprehensive analysis
            }
        }
        
        response = requests.post(url, json=payload, timeout=600)  # 10-minute timeout
        response.raise_for_status()
        
        response_data = response.json()
        return response_data.get("response", "Error: 'response' key not found in Ollama's output.")
    
    except requests.exceptions.ConnectionError:
        logger.error(f"ConnectionError: Could not connect to Ollama at {url}")
        logger.error("Ensure Ollama is running and the model is available")
        return None
    except requests.exceptions.Timeout:
        logger.error("Request timed out. The analysis may be too complex for the current model.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Ollama: {e}")
        return None


def save_analysis_results(summary: str, ai_report: Optional[str], df: pd.DataFrame, output_dir: str = "analysis_results") -> None:
    """Save comprehensive analysis results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    size_suffix = f"{len(df):,}".replace(",", "")
    
    # Save statistical summary
    summary_file = os.path.join(output_dir, f"mathematical_analysis_{size_suffix}_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    # Save AI report if available
    if ai_report:
        ai_file = os.path.join(output_dir, f"ai_analysis_{size_suffix}_{timestamp}.txt")
        with open(ai_file, 'w') as f:
            f.write(ai_report)
    
    logger.info(f"Analysis results saved to {output_dir}/")


def main():
    """Enhanced main execution with comprehensive mathematical analysis."""
    parser = argparse.ArgumentParser(
        description='Advanced Collatz CSV analyzer with mathematical intelligence',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('csv_file', type=str, help='Path to Collatz CSV data file')
    parser.add_argument('--model', type=str, default='llama3.1:70b', 
                       help='Ollama model (recommended: llama3.1:70b or deepseek-math:7b)')
    parser.add_argument('--url', type=str, default='http://localhost:11434/api/generate',
                       help='Ollama API endpoint')
    parser.add_argument('--top-n', type=int, default=None,
                       help='Number of outliers to analyze (auto-selected based on dataset size if not specified)')
    parser.add_argument('--save-results', action='store_true',
                       help='Save analysis results to files')
    parser.add_argument('--skip-ai', action='store_true',
                       help='Skip AI analysis and only perform statistical analysis')
    
    args = parser.parse_args()
    
    # Load and validate data
    try:
        logger.info(f"Loading Collatz data from {args.csv_file}...")
        df = pd.read_csv(args.csv_file)
        logger.info(f"Successfully loaded {len(df):,} trajectories")
    except FileNotFoundError:
        logger.error(f"File not found: {args.csv_file}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Initialize mathematical analyzer
    try:
        analyzer = MathematicalAnalyzer(df)
    except ValueError as e:
        logger.error(f"Data validation failed: {e}")
        sys.exit(1)
    
    # Determine outlier analysis size
    top_n = args.top_n if args.top_n else intelligent_outlier_selection(df)
    logger.info(f"Analyzing top {top_n} outliers in each category")
    
    # Generate comprehensive mathematical summary
    logger.info("Performing comprehensive mathematical analysis...")
    summary = generate_comprehensive_summary(analyzer, top_n)
    
    # Display statistical analysis
    print("\n" + "="*80)
    print(" MATHEMATICAL ANALYSIS REPORT")
    print("="*80)
    print(summary)
    
    # Generate AI analysis if requested
    ai_report = None
    if not args.skip_ai:
        ai_report = generate_ollama_report(summary, analyzer, args.model, args.url)
        
        if ai_report:
            print("\n" + "="*80)
            print(f" AI MATHEMATICAL ANALYSIS (Model: {args.model})")
            print("="*80)
            print(ai_report)
        else:
            logger.warning("AI analysis failed. Statistical analysis complete.")
    
    # Save results if requested
    if args.save_results:
        save_analysis_results(summary, ai_report, df)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
