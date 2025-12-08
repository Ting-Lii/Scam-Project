"""
OpenAI ChatGPT API client wrapper
"""

import re
import json
import logging
import openai
from typing import Dict, Any, List

from .prompt_scam_analysis import get_scam_analysis_prompt

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API interactions"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key
            model_name: Model name to use (default: gpt-4o, alternatives: gpt-4, gpt-3.5-turbo)
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def analyze_text(self, text: str, complaint_id: str = None) -> Dict[str, Any]:
        """
        Use ChatGPT to analyze text for scam indicators
        
        Args:
            text: Text to analyze
            complaint_id: Optional complaint ID for tracking
            
        Returns:
            Analysis results from ChatGPT
        """
        # Get prompt from prompts module
        prompt = get_scam_analysis_prompt(text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing job-related complaints for scam indicators. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                response_format={"type": "json_object"}  # Request JSON format
            )
            
            result_text = response.choices[0].message.content
            
            # Try to parse JSON from response
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON from response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    # Fallback parsing
                    return self._parse_fallback_response(result_text)
                
        except Exception as e:
            logger.error(f"Error analyzing with OpenAI: {e}")
            return {
                "scam_probability": 0,
                "red_flags": [],
                "financial_risk": "Unknown",
                "scam_type": "Unknown",
                "victim_profile": "Unknown",
                "recommendations": [],
                "confidence": 0,
                "error": str(e)
            }
    
    def _parse_fallback_response(self, text: str) -> Dict[str, Any]:
        """Parse response when JSON extraction fails"""
        return {
            "scam_probability": self._extract_score(text, "scam_probability"),
            "red_flags": self._extract_list(text, "red_flags"),
            "financial_risk": self._extract_text(text, "financial_risk"),
            "scam_type": self._extract_text(text, "scam_type"),
            "victim_profile": self._extract_text(text, "victim_profile"),
            "recommendations": self._extract_list(text, "recommendations"),
            "confidence": self._extract_score(text, "confidence")
        }
    
    def _extract_score(self, text: str, field: str) -> int:
        """Extract numeric score from text"""
        pattern = rf'{field}["\s]*:?\s*(\d+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else 0
    
    def _extract_text(self, text: str, field: str) -> str:
        """Extract text field from response"""
        pattern = rf'{field}["\s]*:?\s*["\']?([^"\',\n]+)["\']?'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else "Unknown"
    
    def _extract_list(self, text: str, field: str) -> List[str]:
        """Extract list field from response"""
        pattern = rf'{field}["\s]*:?\s*\[(.*?)\]'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            items = re.findall(r'"([^"]+)"', match.group(1))
            return items
        return []

