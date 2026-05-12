"""
Compute attribute-demographic wise match rates for face recognition evaluation.

This script takes pre-computed similarity scores and a threshold, then generates
a table showing the True Match Rate (TMR) for each attribute-demographic combination.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np


DEMOGRAPHICS = [
    'asian_man', 'asian_woman',
    'black_man', 'black_woman',
    'white_man', 'white_woman',
    'indian_man', 'indian_woman'
]

DEMO_SHORT_NAMES = {
    'asian_man': 'AM', 'asian_woman': 'AF',
    'black_man': 'BM', 'black_woman': 'BF',
    'white_man': 'WM', 'white_woman': 'WF',
    'indian_man': 'IM', 'indian_woman': 'IF'
}


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def compute_match_rates(
    similarity_scores: dict,
    paths_attr_demo_wise: dict,
    threshold: float,
    path_transform_fn=None
) -> dict:
    """
    Compute match rates for each attribute-demographic combination.
    
    Args:
        similarity_scores: Dict mapping "path1|path2" to similarity score
        paths_attr_demo_wise: Dict[attr][demo] -> list of [path1, path2] pairs
        threshold: Similarity threshold for a match
        path_transform_fn: Optional function to transform image paths to score keys
    
    Returns:
        Dict[attr][demo] -> match rate (0.0 to 1.0)
    """
    attr_demo_match_rates = {}
    
    for attr in paths_attr_demo_wise:
        attr_demo_match_rates[attr] = {}
        for demo in paths_attr_demo_wise[attr]:
            paths = paths_attr_demo_wise[attr][demo]
            
            # Handle nested list format (some datasets have [[path1, path2], ...])
            if paths and isinstance(paths[0], str):
                # Single level: [path1, path2, path3, ...]
                # Assume pairs are stored as consecutive elements
                pass
            
            matches = 0
            total = 0
            
            for pair in paths:
                if isinstance(pair, list) and len(pair) >= 2:
                    path1, path2 = pair[0], pair[1]
                else:
                    continue
                
                # Apply path transformation if provided
                if path_transform_fn:
                    key = path_transform_fn(path1, path2)
                else:
                    key = f"{path1}|{path2}"
                
                if key in similarity_scores:
                    sim_score = similarity_scores[key]
                    if isinstance(sim_score, (int, float)):
                        total += 1
                        if sim_score >= threshold:
                            matches += 1
            
            if total > 0:
                attr_demo_match_rates[attr][demo] = matches / total
            else:
                attr_demo_match_rates[attr][demo] = None
    
    return attr_demo_match_rates


def format_results_dataframe(
    match_rates: dict,
    use_short_names: bool = True,
    multiply_by_100: bool = True
) -> pd.DataFrame:
    """Convert match rates dict to a formatted DataFrame."""
    df = pd.DataFrame(match_rates).T
    
    # Reorder columns to standard demographic order
    cols = [d for d in DEMOGRAPHICS if d in df.columns]
    df = df[cols]
    
    if use_short_names:
        df.rename(columns=DEMO_SHORT_NAMES, inplace=True)
    
    if multiply_by_100:
        df = df * 100
    
    # Clean up index (attribute names)
    df.index = df.index.str.replace('_', ' ')
    
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute attribute-demographic wise match rates'
    )
    parser.add_argument(
        '--similarity_scores', type=str, required=True,
        help='Path to JSON file with similarity scores (key format: "path1|path2")'
    )
    parser.add_argument(
        '--paths_json', type=str, required=True,
        help='Path to JSON file with attribute-demo wise path pairs'
    )
    parser.add_argument(
        '--threshold', type=float, required=True,
        help='Similarity threshold for match'
    )
    parser.add_argument(
        '--output_csv', type=str, default=None,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--output_latex', type=str, default=None,
        help='Output LaTeX table file path'
    )
    parser.add_argument(
        '--output_json', type=str, default=None,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--raw_values', action='store_true',
        help='Output raw values (0-1) instead of percentages (0-100)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading similarity scores from {args.similarity_scores}...")
    similarity_scores = load_json(args.similarity_scores)
    print(f"  Loaded {len(similarity_scores)} score entries")
    
    print(f"Loading paths from {args.paths_json}...")
    paths_attr_demo_wise = load_json(args.paths_json)
    num_attrs = len(paths_attr_demo_wise)
    print(f"  Loaded {num_attrs} attributes")
    
    print(f"\nComputing match rates with threshold={args.threshold}...")
    match_rates = compute_match_rates(
        similarity_scores,
        paths_attr_demo_wise,
        args.threshold
    )
    
    # Create DataFrame
    multiply = not args.raw_values
    df = format_results_dataframe(match_rates, multiply_by_100=multiply)
    
    # Print summary
    print("\n" + "="*60)
    print("Match Rates by Attribute and Demographic")
    print("="*60)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df.to_string())
    
    # Compute overall statistics
    values = df.values.flatten()
    valid_values = [v for v in values if pd.notna(v)]
    if valid_values:
        print(f"\nOverall Mean: {np.mean(valid_values):.2f}")
        print(f"Overall Std:  {np.std(valid_values):.2f}")
        print(f"Min:          {np.min(valid_values):.2f}")
        print(f"Max:          {np.max(valid_values):.2f}")
    
    # Save outputs
    if args.output_csv:
        df.to_csv(args.output_csv)
        print(f"\nSaved CSV to {args.output_csv}")
    
    if args.output_latex:
        latex_str = df.to_latex(float_format="%.2f")
        with open(args.output_latex, 'w') as f:
            f.write(latex_str)
        print(f"Saved LaTeX to {args.output_latex}")
    
    if args.output_json:
        save_json(match_rates, args.output_json)
        print(f"Saved JSON to {args.output_json}")


if __name__ == '__main__':
    main()
