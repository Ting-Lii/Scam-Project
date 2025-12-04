#!/usr/bin/env python3
"""
Chain of Thought (CoT) Comparison Script
Compares detection accuracy and reasoning quality between standard prompts and Chain of Thought prompts
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

from scam_detector.text_processor import TextProcessor
from scam_detector.gemini_client import GeminiClient
from scam_detector.openai_client import OpenAIClient
from scam_detector.file_handler import FileHandler
from scam_detector.prompt_scam_analysis import get_scam_analysis_prompt
from scam_detector.prompt_scam_analysis_cot import get_scam_analysis_prompt_cot

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoTComparator:
    """Compare detection results between standard prompts and Chain of Thought prompts"""
    
    def __init__(self, gemini_api_key: str = None, openai_api_key: str = None, 
                 openai_model: str = "gpt-4o", output_dir: Path = None, use_openai: bool = True):
        """
        Initialize CoT Comparator
        
        Args:
            gemini_api_key: Google Gemini API key
            openai_api_key: OpenAI API key
            openai_model: OpenAI model to use (default: gpt-4o)
            output_dir: Output directory for comparison results
            use_openai: Whether to use OpenAI in addition to Gemini (default: True)
        """
        # Get API keys from environment if not provided
        if gemini_api_key is None:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        if openai_api_key is None:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        if use_openai and not openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.gemini_client = GeminiClient(gemini_api_key)
        self.use_openai = use_openai
        if use_openai:
            self.openai_client = OpenAIClient(openai_api_key, openai_model)
        self.file_handler = FileHandler(output_dir)
        self.openai_model = openai_model
    
    def analyze_with_standard_prompt(self, complaint_text: str, complaint_id: str = None, 
                                     use_gemini: bool = True, use_openai: bool = True) -> Dict[str, Any]:
        """
        Analyze a complaint using standard (non-CoT) prompt
        
        Args:
            complaint_text: The complaint text to analyze
            complaint_id: Optional complaint ID
            use_gemini: Whether to use Gemini
            use_openai: Whether to use OpenAI
            
        Returns:
            Analysis results from standard prompt
        """
        cleaned_text = self.text_processor.preprocess_text(complaint_text)
        results = {}
        
        if use_gemini:
            logger.info(f"Analyzing with Gemini (Standard): {complaint_id}")
            results['gemini'] = self.gemini_client.analyze_text(cleaned_text, complaint_id, prompt_mode='job_scam')
        
        if use_openai and self.use_openai:
            logger.info(f"Analyzing with OpenAI (Standard): {complaint_id}")
            # Temporarily modify OpenAI client to use standard prompt
            original_prompt = get_scam_analysis_prompt(cleaned_text)
            # OpenAI client uses get_scam_analysis_prompt internally, so we need to patch it
            # For now, we'll use the client as-is since it already uses standard prompt
            results['openai'] = self.openai_client.analyze_text(cleaned_text, complaint_id)
        
        return results
    
    def analyze_with_cot_prompt(self, complaint_text: str, complaint_id: str = None,
                                use_gemini: bool = True, use_openai: bool = True) -> Dict[str, Any]:
        """
        Analyze a complaint using Chain of Thought prompt
        
        Args:
            complaint_text: The complaint text to analyze
            complaint_id: Optional complaint ID
            use_gemini: Whether to use Gemini
            use_openai: Whether to use OpenAI
            
        Returns:
            Analysis results from CoT prompt
        """
        cleaned_text = self.text_processor.preprocess_text(complaint_text)
        results = {}
        
        if use_gemini:
            logger.info(f"Analyzing with Gemini (CoT): {complaint_id}")
            # Use CoT prompt directly
            cot_prompt = get_scam_analysis_prompt_cot(cleaned_text)
            # Call the model directly with the CoT prompt, using the client's rate limiter
            try:
                # Apply rate limiting if enabled
                if self.gemini_client.rate_limiter:
                    self.gemini_client.rate_limiter.wait_if_needed()
                
                response = self.gemini_client.model.generate_content(cot_prompt)
                result_text = response.text
                
                # Extract JSON from response (using same logic as GeminiClient)
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1) if json_match.lastindex else json_match.group()
                    try:
                        results['gemini'] = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to fix JSON using client's method
                        fixed_json = self.gemini_client._fix_json_string(json_str)
                        try:
                            results['gemini'] = json.loads(fixed_json)
                        except json.JSONDecodeError:
                            results['gemini'] = self.gemini_client._parse_fallback_response(result_text)
                else:
                    results['gemini'] = self.gemini_client._parse_fallback_response(result_text)
            except Exception as e:
                logger.error(f"Error in Gemini CoT analysis: {e}")
                results['gemini'] = {
                    "scam_probability": 0,
                    "red_flags": {},
                    "financial_risk": {"level": "Unknown"},
                    "scam_type": {"primary_category": "Unknown", "subcategory": "Unknown"},
                    "victim_profile": {"risk_level": "Unknown"},
                    "recommendations": {},
                    "confidence": 0,
                    "error": str(e)
                }
        
        if use_openai and self.use_openai:
            logger.info(f"Analyzing with OpenAI (CoT): {complaint_id}")
            # Use CoT prompt with OpenAI
            cot_prompt = get_scam_analysis_prompt_cot(cleaned_text)
            try:
                response = self.openai_client.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing job-related complaints for scam indicators. Always respond with valid JSON."},
                        {"role": "user", "content": cot_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                result_text = response.choices[0].message.content
                results['openai'] = json.loads(result_text)
            except Exception as e:
                logger.error(f"Error in OpenAI CoT analysis: {e}")
                results['openai'] = {
                    "scam_probability": 0,
                    "error": str(e)
                }
        
        return results
    
    def _count_red_flags(self, red_flags: Any) -> int:
        """
        Count total number of red flags across all categories
        
        Args:
            red_flags: Red flags dict or list
            
        Returns:
            Total count of red flags
        """
        if not red_flags:
            return 0
        
        if isinstance(red_flags, dict):
            total = 0
            for category in ['communication', 'financial', 'job_posting', 'hiring_process', 'work_activity']:
                flags = red_flags.get(category, [])
                if isinstance(flags, list):
                    total += len(flags)
            return total
        elif isinstance(red_flags, list):
            return len(red_flags)
        else:
            return 0
    
    def compare_prompts(self, complaint_text: str, complaint_id: str = None) -> Dict[str, Any]:
        """
        Compare standard vs CoT prompts on the same complaint
        
        Args:
            complaint_text: The complaint text to analyze
            complaint_id: Optional complaint ID
            
        Returns:
            Comparison results
        """
        # Analyze with standard prompt
        standard_results = self.analyze_with_standard_prompt(complaint_text, complaint_id)
        
        # Analyze with CoT prompt
        cot_results = self.analyze_with_cot_prompt(complaint_text, complaint_id)
        
        # Compare results
        comparison = {
            'complaint_id': complaint_id,
            'timestamp': datetime.now().isoformat(),
            'text_length': len(self.text_processor.preprocess_text(complaint_text)),
            'standard': standard_results,
            'cot': cot_results,
        }
        
        # Add comparison metrics
        if 'gemini' in standard_results and 'gemini' in cot_results:
            std_score = standard_results['gemini'].get('scam_probability', 0)
            cot_score = cot_results['gemini'].get('scam_probability', 0)
            comparison['gemini'] = {
                'standard_score': std_score,
                'cot_score': cot_score,
                'score_difference': abs(std_score - cot_score),
                'score_change': cot_score - std_score,
                'standard_confidence': standard_results['gemini'].get('confidence', 0),
                'cot_confidence': cot_results['gemini'].get('confidence', 0),
                'has_reasoning': 'reasoning_steps' in cot_results['gemini'],
                'red_flags_std': self._count_red_flags(standard_results['gemini'].get('red_flags', {})),
                'red_flags_cot': self._count_red_flags(cot_results['gemini'].get('red_flags', {}))
            }
        
        if 'openai' in standard_results and 'openai' in cot_results:
            std_score = standard_results['openai'].get('scam_probability', 0)
            cot_score = cot_results['openai'].get('scam_probability', 0)
            comparison['openai'] = {
                'standard_score': std_score,
                'cot_score': cot_score,
                'score_difference': abs(std_score - cot_score),
                'score_change': cot_score - std_score,
                'standard_confidence': standard_results['openai'].get('confidence', 0),
                'cot_confidence': cot_results['openai'].get('confidence', 0),
                'has_reasoning': 'reasoning_steps' in cot_results['openai'],
                'red_flags_std': self._count_red_flags(standard_results['openai'].get('red_flags', {})),
                'red_flags_cot': self._count_red_flags(cot_results['openai'].get('red_flags', {}))
            }
        
        return comparison
    
    def compare_datasets(self, csv_file_path: str, output_filename: str = None, 
                        max_complaints: int = None, use_gemini: bool = True, 
                        use_openai: bool = True) -> pd.DataFrame:
        """
        Compare standard vs CoT prompts on a dataset
        
        Args:
            csv_file_path: Path to CSV file with complaints
            output_filename: Optional output filename
            max_complaints: Optional limit on number of complaints to analyze
            use_gemini: Whether to use Gemini for comparison
            use_openai: Whether to use OpenAI for comparison
            
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
        
        logger.info(f"Comparing Standard vs CoT prompts on {total} complaints...")
        
        for idx, row in df.iterrows():
            try:
                complaint_text = row['Consumer complaint narrative']
                complaint_id = row[complaint_id_col] if complaint_id_col else f"complaint_{idx}"
                
                logger.info(f"Processing complaint {idx + 1}/{total}: {complaint_id}")
                
                comparison = self.compare_prompts(complaint_text, complaint_id)
                results.append(comparison)
                
            except Exception as e:
                logger.error(f"Error analyzing complaint {idx}: {e}")
                continue
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Flatten nested dictionaries for CSV export
        flattened_results = []
        for _, row in comparison_df.iterrows():
            flat_row = {
                'complaint_id': row['complaint_id'],
                'timestamp': row['timestamp'],
                'text_length': row['text_length'],
            }
            
            # Add Gemini metrics if available
            if 'gemini' in row and pd.notna(row['gemini']):
                gemini_metrics = row['gemini']
                flat_row.update({
                    'gemini_standard_score': gemini_metrics.get('standard_score', 0),
                    'gemini_cot_score': gemini_metrics.get('cot_score', 0),
                    'gemini_score_difference': gemini_metrics.get('score_difference', 0),
                    'gemini_score_change': gemini_metrics.get('score_change', 0),
                    'gemini_standard_confidence': gemini_metrics.get('standard_confidence', 0),
                    'gemini_cot_confidence': gemini_metrics.get('cot_confidence', 0),
                    'gemini_has_reasoning': gemini_metrics.get('has_reasoning', False),
                    'gemini_red_flags_std': gemini_metrics.get('red_flags_std', 0),
                    'gemini_red_flags_cot': gemini_metrics.get('red_flags_cot', 0),
                })
            
            # Add OpenAI metrics if available
            if 'openai' in row and pd.notna(row['openai']):
                openai_metrics = row['openai']
                flat_row.update({
                    'openai_standard_score': openai_metrics.get('standard_score', 0),
                    'openai_cot_score': openai_metrics.get('cot_score', 0),
                    'openai_score_difference': openai_metrics.get('score_difference', 0),
                    'openai_score_change': openai_metrics.get('score_change', 0),
                    'openai_standard_confidence': openai_metrics.get('standard_confidence', 0),
                    'openai_cot_confidence': openai_metrics.get('cot_confidence', 0),
                    'openai_has_reasoning': openai_metrics.get('has_reasoning', False),
                    'openai_red_flags_std': openai_metrics.get('red_flags_std', 0),
                    'openai_red_flags_cot': openai_metrics.get('red_flags_cot', 0),
                })
            
            flattened_results.append(flat_row)
        
        comparison_df_flat = pd.DataFrame(flattened_results)
        
        # Save results
        if output_filename is None:
            output_filename = f"cot_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        output_path = self.file_handler.save_results_csv(comparison_df_flat, output_filename)
        logger.info(f"Comparison results saved to {output_path}")
        
        return comparison_df_flat
    
    def _convert_to_native_types(self, obj: Any) -> Any:
        """
        Convert numpy/pandas types to native Python types for JSON serialization
        
        Args:
            obj: Object that may contain numpy/pandas types
            
        Returns:
            Object with native Python types
        """
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native_types(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def generate_comparison_report(self, comparison_df: pd.DataFrame, 
                                  report_filename: str = None) -> Dict[str, Any]:
        """
        Generate comparison report with metrics
        
        Args:
            comparison_df: DataFrame with comparison results
            report_filename: Optional report filename
            
        Returns:
            Comparison report dictionary
        """
        total = len(comparison_df)
        
        report = {
            'summary': {
                'total_complaints': int(total),
                'timestamp': datetime.now().isoformat()
            },
            'gemini_metrics': {},
            'openai_metrics': {},
            'model_info': {
                'gemini_model': self.gemini_client.model_name,
                'openai_model': self.openai_model if self.use_openai else None
            }
        }
        
        # Gemini metrics
        if 'gemini_standard_score' in comparison_df.columns:
            report['gemini_metrics'] = {
                'avg_standard_score': float(comparison_df['gemini_standard_score'].mean()),
                'avg_cot_score': float(comparison_df['gemini_cot_score'].mean()),
                'avg_score_difference': float(comparison_df['gemini_score_difference'].mean()),
                'avg_score_change': float(comparison_df['gemini_score_change'].mean()),
                'avg_standard_confidence': float(comparison_df['gemini_standard_confidence'].mean()),
                'avg_cot_confidence': float(comparison_df['gemini_cot_confidence'].mean()),
                'avg_red_flags_std': float(comparison_df['gemini_red_flags_std'].mean()),
                'avg_red_flags_cot': float(comparison_df['gemini_red_flags_cot'].mean()),
                'has_reasoning_count': int(comparison_df['gemini_has_reasoning'].sum() if 'gemini_has_reasoning' in comparison_df.columns else 0),
                'score_increased': int((comparison_df['gemini_score_change'] > 0).sum()),
                'score_decreased': int((comparison_df['gemini_score_change'] < 0).sum()),
                'score_unchanged': int((comparison_df['gemini_score_change'] == 0).sum()),
            }
        
        # OpenAI metrics
        if 'openai_standard_score' in comparison_df.columns:
            report['openai_metrics'] = {
                'avg_standard_score': float(comparison_df['openai_standard_score'].mean()),
                'avg_cot_score': float(comparison_df['openai_cot_score'].mean()),
                'avg_score_difference': float(comparison_df['openai_score_difference'].mean()),
                'avg_score_change': float(comparison_df['openai_score_change'].mean()),
                'avg_standard_confidence': float(comparison_df['openai_standard_confidence'].mean()),
                'avg_cot_confidence': float(comparison_df['openai_cot_confidence'].mean()),
                'avg_red_flags_std': float(comparison_df['openai_red_flags_std'].mean()),
                'avg_red_flags_cot': float(comparison_df['openai_red_flags_cot'].mean()),
                'has_reasoning_count': int(comparison_df['openai_has_reasoning'].sum() if 'openai_has_reasoning' in comparison_df.columns else 0),
                'score_increased': int((comparison_df['openai_score_change'] > 0).sum()),
                'score_decreased': int((comparison_df['openai_score_change'] < 0).sum()),
                'score_unchanged': int((comparison_df['openai_score_change'] == 0).sum()),
            }
        
        # Convert all numpy/pandas types to native Python types
        report = self._convert_to_native_types(report)
        
        # Save report
        if report_filename is None:
            report_filename = f"cot_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = self.file_handler.output_dir / report_filename
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comparison report saved to {report_path}")
        
        return report


def main():
    """Main function to run CoT comparison"""
    
    parser = argparse.ArgumentParser(
        description='Compare detection accuracy between standard prompts and Chain of Thought prompts'
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
        help='OpenAI model to use (default: gpt-4o)'
    )
    parser.add_argument(
        '--max-complaints',
        type=int,
        default=None,
        help='Maximum number of complaints to analyze (for testing)'
    )
    parser.add_argument(
        '--gemini-only',
        action='store_true',
        help='Only use Gemini (skip OpenAI)'
    )
    parser.add_argument(
        '--openai-only',
        action='store_true',
        help='Only use OpenAI (skip Gemini)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine which models to use
    use_gemini = not args.openai_only
    use_openai = not args.gemini_only
    
    # Initialize comparator
    output_dir = Path(args.output_dir) if args.output_dir else None
    try:
        comparator = CoTComparator(
            gemini_api_key=args.gemini_api_key,
            openai_api_key=args.openai_api_key,
            openai_model=args.openai_model,
            output_dir=output_dir,
            use_openai=use_openai
        )
    except Exception as e:
        print(f"Error initializing comparator: {e}")
        sys.exit(1)
    
    try:
        # Compare datasets
        logger.info(f"Starting CoT comparison on {input_path}")
        comparison_df = comparator.compare_datasets(
            str(input_path),
            max_complaints=args.max_complaints,
            use_gemini=use_gemini,
            use_openai=use_openai
        )
        
        # Generate comparison report
        logger.info("Generating comparison report...")
        report = comparator.generate_comparison_report(comparison_df)
        
        # Print summary
        print("\n" + "="*60)
        print("CHAIN OF THOUGHT COMPARISON REPORT")
        print("="*60)
        print(f"Total Complaints Analyzed: {report['summary']['total_complaints']}")
        
        if report['gemini_metrics']:
            print(f"\n{'='*60}")
            print("GEMINI METRICS")
            print(f"{'='*60}")
            gm = report['gemini_metrics']
            print(f"Average Scores:")
            print(f"  Standard Prompt: {gm['avg_standard_score']:.1f}%")
            print(f"  CoT Prompt: {gm['avg_cot_score']:.1f}%")
            print(f"  Average Difference: {gm['avg_score_difference']:.1f}%")
            print(f"  Average Change: {gm['avg_score_change']:+.1f}%")
            print(f"\nConfidence Levels:")
            print(f"  Standard Prompt: {gm['avg_standard_confidence']:.1f}%")
            print(f"  CoT Prompt: {gm['avg_cot_confidence']:.1f}%")
            print(f"\nRed Flags Detected:")
            print(f"  Standard Prompt: {gm['avg_red_flags_std']:.1f} flags")
            print(f"  CoT Prompt: {gm['avg_red_flags_cot']:.1f} flags")
            print(f"\nScore Changes:")
            print(f"  Increased: {gm['score_increased']} cases")
            print(f"  Decreased: {gm['score_decreased']} cases")
            print(f"  Unchanged: {gm['score_unchanged']} cases")
            print(f"  Has Reasoning Steps: {gm['has_reasoning_count']} cases")
        
        if report['openai_metrics']:
            print(f"\n{'='*60}")
            print("OPENAI METRICS")
            print(f"{'='*60}")
            om = report['openai_metrics']
            print(f"Average Scores:")
            print(f"  Standard Prompt: {om['avg_standard_score']:.1f}%")
            print(f"  CoT Prompt: {om['avg_cot_score']:.1f}%")
            print(f"  Average Difference: {om['avg_score_difference']:.1f}%")
            print(f"  Average Change: {om['avg_score_change']:+.1f}%")
            print(f"\nConfidence Levels:")
            print(f"  Standard Prompt: {om['avg_standard_confidence']:.1f}%")
            print(f"  CoT Prompt: {om['avg_cot_confidence']:.1f}%")
            print(f"\nRed Flags Detected:")
            print(f"  Standard Prompt: {om['avg_red_flags_std']:.1f} flags")
            print(f"  CoT Prompt: {om['avg_red_flags_cot']:.1f} flags")
            print(f"\nScore Changes:")
            print(f"  Increased: {om['score_increased']} cases")
            print(f"  Decreased: {om['score_decreased']} cases")
            print(f"  Unchanged: {om['score_unchanged']} cases")
            print(f"  Has Reasoning Steps: {om['has_reasoning_count']} cases")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

