#!/usr/bin/env python3
"""
LLM Comparison Script
Compares detection accuracy between Gemini and ChatGPT on the same dataset
"""

import os
import sys
import logging
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from scam_detector.text_processor import TextProcessor
from scam_detector.gemini_client import GeminiClient
from scam_detector.openai_client import OpenAIClient
from scam_detector.file_handler import FileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMComparator:
    """Compare detection results between different LLMs"""
    
    def __init__(self, gemini_api_key: str = None, openai_api_key: str = None, 
                 openai_model: str = "gpt-4o", output_dir: Path = None):
        """
        Initialize LLM Comparator
        
        Args:
            gemini_api_key: Google Gemini API key
            openai_api_key: OpenAI API key
            openai_model: OpenAI model to use (default: gpt-4o)
            output_dir: Output directory for comparison results
        """
        # Get API keys from environment if not provided
        if gemini_api_key is None:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        if openai_api_key is None:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.gemini_client = GeminiClient(gemini_api_key)
        self.openai_client = OpenAIClient(openai_api_key, openai_model)
        self.file_handler = FileHandler(output_dir)
        self.openai_model = openai_model
    
    def analyze_complaint_both_llms(self, complaint_text: str, complaint_id: str = None) -> Dict[str, Any]:
        """
        Analyze a complaint using both LLMs
        
        Args:
            complaint_text: The complaint text to analyze
            complaint_id: Optional complaint ID
            
        Returns:
            Combined analysis results from both LLMs
        """
        # Preprocess text
        cleaned_text = self.text_processor.preprocess_text(complaint_text)
        
        # Get analysis from both LLMs
        logger.info(f"Analyzing with Gemini: {complaint_id}")
        gemini_analysis = self.gemini_client.analyze_text(cleaned_text, complaint_id)
        
        logger.info(f"Analyzing with ChatGPT: {complaint_id}")
        openai_analysis = self.openai_client.analyze_text(cleaned_text, complaint_id)
        
        # Combine results
        analysis = {
            'complaint_id': complaint_id,
            'timestamp': datetime.now().isoformat(),
            'text_length': len(cleaned_text),
            'gemini_analysis': gemini_analysis,
            'openai_analysis': openai_analysis,
            'gemini_risk_score': gemini_analysis.get('scam_probability', 0),
            'openai_risk_score': openai_analysis.get('scam_probability', 0),
            'risk_score_difference': abs(gemini_analysis.get('scam_probability', 0) - 
                                        openai_analysis.get('scam_probability', 0)),
            'agreement': self._check_agreement(gemini_analysis, openai_analysis)
        }
        
        return analysis
    
    def _check_agreement(self, gemini_analysis: Dict, openai_analysis: Dict) -> Dict[str, Any]:
        """
        Check agreement between two LLM analyses
        
        Args:
            gemini_analysis: Gemini analysis results
            openai_analysis: OpenAI analysis results
            
        Returns:
            Agreement metrics
        """
        gemini_score = gemini_analysis.get('scam_probability', 0)
        openai_score = openai_analysis.get('scam_probability', 0)
        
        # Define thresholds
        high_risk_threshold = 70
        medium_risk_threshold = 40
        
        gemini_category = self._categorize_risk(gemini_score, high_risk_threshold, medium_risk_threshold)
        openai_category = self._categorize_risk(openai_score, high_risk_threshold, medium_risk_threshold)
        
        return {
            'score_difference': abs(gemini_score - openai_score),
            'category_match': gemini_category == openai_category,
            'gemini_category': gemini_category,
            'openai_category': openai_category,
            'both_high_risk': gemini_category == 'High' and openai_category == 'High',
            'both_low_risk': gemini_category == 'Low' and openai_category == 'Low'
        }
    
    def _categorize_risk(self, score: int, high_threshold: int, medium_threshold: int) -> str:
        """Categorize risk score"""
        if score >= high_threshold:
            return 'High'
        elif score >= medium_threshold:
            return 'Medium'
        else:
            return 'Low'
    
    def compare_datasets(self, csv_file_path: str, output_filename: str = None, 
                        max_complaints: int = None) -> pd.DataFrame:
        """
        Compare both LLMs on a dataset
        
        Args:
            csv_file_path: Path to CSV file with complaints
            output_filename: Optional output filename
            max_complaints: Optional limit on number of complaints to analyze
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Loading dataset from {csv_file_path}")
        
        # Load data
        df = pd.read_csv(csv_file_path)
        
        # Ensure we have the required columns
        if 'Consumer complaint narrative' not in df.columns:
            raise ValueError("CSV must contain 'Consumer complaint narrative' column")
        
        complaint_id_col = 'Complaint ID' if 'Complaint ID' in df.columns else None
        
        # Limit complaints if specified
        if max_complaints:
            df = df.head(max_complaints)
            logger.info(f"Limited to first {max_complaints} complaints for comparison")
        
        results = []
        total = len(df)
        
        logger.info(f"Comparing LLMs on {total} complaints...")
        
        for idx, row in df.iterrows():
            try:
                complaint_text = row['Consumer complaint narrative']
                complaint_id = row[complaint_id_col] if complaint_id_col else f"complaint_{idx}"
                
                logger.info(f"Processing complaint {idx + 1}/{total}: {complaint_id}")
                
                analysis = self.analyze_complaint_both_llms(complaint_text, complaint_id)
                results.append(analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing complaint {idx}: {e}")
                continue
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Flatten nested dictionaries for CSV export
        flattened_results = []
        for _, row in results_df.iterrows():
            flat_row = {
                'complaint_id': row['complaint_id'],
                'timestamp': row['timestamp'],
                'text_length': row['text_length'],
                'gemini_risk_score': row['gemini_risk_score'],
                'openai_risk_score': row['openai_risk_score'],
                'risk_score_difference': row['risk_score_difference'],
                'category_match': row['agreement']['category_match'],
                'gemini_category': row['agreement']['gemini_category'],
                'openai_category': row['agreement']['openai_category'],
                'both_high_risk': row['agreement']['both_high_risk'],
                'both_low_risk': row['agreement']['both_low_risk'],
                'gemini_red_flags': ', '.join(row['gemini_analysis'].get('red_flags', [])),
                'openai_red_flags': ', '.join(row['openai_analysis'].get('red_flags', [])),
                'gemini_scam_type': row['gemini_analysis'].get('scam_type', 'Unknown'),
                'openai_scam_type': row['openai_analysis'].get('scam_type', 'Unknown'),
                'gemini_financial_risk': row['gemini_analysis'].get('financial_risk', 'Unknown'),
                'openai_financial_risk': row['openai_analysis'].get('financial_risk', 'Unknown'),
            }
            flattened_results.append(flat_row)
        
        comparison_df = pd.DataFrame(flattened_results)
        
        # Save results
        if output_filename is None:
            output_filename = f"llm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = self.file_handler.save_results_csv(comparison_df, output_filename)
        logger.info(f"Comparison results saved to {output_path}")
        
        return comparison_df
    
    def generate_comparison_report(self, comparison_df: pd.DataFrame, 
                                  report_filename: str = None) -> Dict[str, Any]:
        """
        Generate comparison report with accuracy metrics
        
        Args:
            comparison_df: DataFrame with comparison results
            report_filename: Optional report filename
            
        Returns:
            Comparison report dictionary
        """
        total = len(comparison_df)
        
        # Calculate agreement metrics
        category_matches = comparison_df['category_match'].sum()
        agreement_rate = (category_matches / total * 100) if total > 0 else 0
        
        # Calculate average scores
        avg_gemini_score = comparison_df['gemini_risk_score'].mean()
        avg_openai_score = comparison_df['openai_risk_score'].mean()
        avg_difference = comparison_df['risk_score_difference'].mean()
        
        # Calculate correlation
        correlation = comparison_df['gemini_risk_score'].corr(comparison_df['openai_risk_score'])
        
        # High risk agreement
        both_high_risk = comparison_df['both_high_risk'].sum()
        both_low_risk = comparison_df['both_low_risk'].sum()
        
        # Count disagreements by category
        disagreements = comparison_df[~comparison_df['category_match']]
        disagreement_breakdown = {}
        if len(disagreements) > 0:
            for _, row in disagreements.iterrows():
                key = f"{row['gemini_category']}_vs_{row['openai_category']}"
                disagreement_breakdown[key] = disagreement_breakdown.get(key, 0) + 1
        
        report = {
            'summary': {
                'total_complaints': total,
                'agreement_rate': agreement_rate,
                'category_matches': category_matches,
                'category_mismatches': total - category_matches,
                'both_high_risk_count': both_high_risk,
                'both_low_risk_count': both_low_risk,
                'average_gemini_score': avg_gemini_score,
                'average_openai_score': avg_openai_score,
                'average_score_difference': avg_difference,
                'correlation': correlation
            },
            'disagreement_breakdown': disagreement_breakdown,
            'model_info': {
                'gemini_model': 'gemini-1.5-pro',
                'openai_model': self.openai_model
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        if report_filename is None:
            report_filename = f"llm_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        report_path = self.file_handler.output_dir / report_filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to {report_path}")
        
        return report


def main():
    """Main function to run LLM comparison"""
    
    parser = argparse.ArgumentParser(
        description='Compare detection accuracy between Gemini and ChatGPT'
    )
    parser.add_argument(
        '--input', 
        type=str,
        required=True,
        help='Path to input CSV file with complaints'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: detect_res/)'
    )
    parser.add_argument(
        '--gemini-api-key',
        type=str,
        default=None,
        help='Google Gemini API key (or set GEMINI_API_KEY env var)'
    )
    parser.add_argument(
        '--openai-api-key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--openai-model',
        type=str,
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o, alternatives: gpt-4, gpt-3.5-turbo)'
    )
    parser.add_argument(
        '--max-complaints',
        type=int,
        default=None,
        help='Maximum number of complaints to analyze (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Initialize comparator
    output_dir = Path(args.output_dir) if args.output_dir else None
    try:
        comparator = LLMComparator(
            gemini_api_key=args.gemini_api_key,
            openai_api_key=args.openai_api_key,
            openai_model=args.openai_model,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"Error initializing comparator: {e}")
        sys.exit(1)
    
    try:
        # Compare datasets
        logger.info(f"Starting comparison on {input_path}")
        comparison_df = comparator.compare_datasets(
            str(input_path),
            max_complaints=args.max_complaints
        )
        
        # Generate comparison report
        logger.info("Generating comparison report...")
        report = comparator.generate_comparison_report(comparison_df)
        
        # Print summary
        print("\n" + "="*60)
        print("LLM COMPARISON REPORT")
        print("="*60)
        print(f"Total Complaints Analyzed: {report['summary']['total_complaints']}")
        print(f"\nAgreement Metrics:")
        print(f"  Category Agreement Rate: {report['summary']['agreement_rate']:.1f}%")
        print(f"  Category Matches: {report['summary']['category_matches']}")
        print(f"  Category Mismatches: {report['summary']['category_mismatches']}")
        print(f"\nAverage Risk Scores:")
        print(f"  Gemini Average: {report['summary']['average_gemini_score']:.1f}%")
        print(f"  ChatGPT Average: {report['summary']['average_openai_score']:.1f}%")
        print(f"  Average Difference: {report['summary']['average_score_difference']:.1f}%")
        print(f"\nCorrelation: {report['summary']['correlation']:.3f}")
        print(f"\nHigh Risk Agreement: {report['summary']['both_high_risk_count']} cases")
        print(f"Low Risk Agreement: {report['summary']['both_low_risk_count']} cases")
        
        if report['disagreement_breakdown']:
            print(f"\nDisagreement Breakdown:")
            for key, count in report['disagreement_breakdown'].items():
                print(f"  {key}: {count} cases")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

