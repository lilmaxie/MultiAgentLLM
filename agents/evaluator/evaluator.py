from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, Template
from utils import call_llm


class EvaluatorAgent:
    def __init__(self, llm, template_dir: str = "agents/evaluator/templates"):
        self.llm = llm
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
        # Load main template
        self.main_template = self.env.get_template("evaluator_main.j2")
        
        # Baseline template mapping
        self.baseline_templates = {
            "health_nutrition": "baselines/health_nutrition.j2",
            "disease_warning": "baselines/disease_warning.j2",
            "travel_adventure": "baselines/travel_adventure.j2",
            "business_enterprise": "baselines/business_enterprise.j2",
            "lifestyle_office": "baselines/lifestyle_office.j2",
            "holiday_event": "baselines/holiday_event.j2",
        }
        
        # Criteria template mapping
        self.criteria_templates = {
            "quality_gate": "criteria/quality_gate.j2",
            "content_information": "criteria/content_information.j2",
            "structure_presentation": "criteria/structure_presentation.j2",
            "affina_connection": "criteria/affina_connection.j2",
            "tone_style": "criteria/tone_style.j2",
            "completeness": "criteria/completeness.j2",
        }
        
        # Default criteria weights
        self.default_criteria = {
            "quality_gate": 0.2,
            "content_information": 0.2,
            "structure_presentation": 0.15,
            "affina_connection": 0.15,
            "tone_style": 0.15,
            "completeness": 0.15,
        }

    def evaluate(self, 
                 candidate: Any,
                 post_type: str = "health_nutrition",
                 language: str = "vietnamese",
                 target_audience: Optional[str] = None,
                 custom_criteria: Optional[Dict[str, float]] = None,
                 evaluation_focus: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate content using Jinja2 templates
        """
        
        # Extract content from candidate
        if isinstance(candidate, dict):
            content_text = candidate.get("content", "")
        else:
            content_text = str(candidate)
        
        # Validate inputs
        if language not in ["vietnamese", "english"]:
            raise ValueError("Language must be 'vietnamese' or 'english'")
            
        if post_type not in self.baseline_templates:
            raise ValueError(f"Post type must be one of: {list(self.baseline_templates.keys())}")
        
        # Merge default criteria with custom criteria
        criteria = self.default_criteria.copy()
        if custom_criteria:
            criteria.update(custom_criteria)
        
        # Normalize criteria weights to sum to 1.0
        total_weight = sum(criteria.values())
        if total_weight > 0:
            criteria = {k: v / total_weight for k, v in criteria.items()}
        
        # Load baseline template
        baseline_template_name = self.baseline_templates[post_type]
        try:
            baseline_template = self.env.get_template(baseline_template_name)
        except Exception as e:
            raise ValueError(f"Failed to load baseline template {baseline_template_name}: {e}")
        
        # Load all criteria templates
        criteria_contents = {}
        for criteria_name, template_name in self.criteria_templates.items():
            try:
                criteria_template = self.env.get_template(template_name)
                criteria_contents[criteria_name] = criteria_template.render(
                    language=language,
                    post_type=post_type,
                    target_audience=target_audience,
                    criteria=criteria,  # Pass criteria to template
                    criteria_weight=criteria.get(criteria_name, 0.0)
                )
            except Exception as e:
                print(f"Warning: Failed to load criteria template {template_name}: {e}")
                criteria_contents[criteria_name] = f"Criteria {criteria_name} template error: {e}"
        
        # Render the baseline template
        template_variables = {
            "language": language,
            "post_type": post_type,
            "target_audience": target_audience,
            "custom_criteria": custom_criteria or {},
            "criteria": criteria,  # Add criteria to template variables
            "evaluation_focus": evaluation_focus,
            "content_text": content_text
        }
        
        try:
            baseline_content = baseline_template.render(**template_variables)
        except Exception as e:
            raise ValueError(f"Failed to render baseline template: {e}")
        
        # Add baseline and criteria content to variables for main template
        template_variables.update({
            "baseline_content": baseline_content,
            "criteria_contents": criteria_contents
        })
        
        # Render main template
        try:
            prompt = self.main_template.render(**template_variables)
        except Exception as e:
            raise ValueError(f"Failed to render main template: {e}")
        
        # Get LLM response
        try:
            raw_response = call_llm(self.llm, prompt).strip()
        except Exception as e:
            raise ValueError(f"Failed to call LLM: {e}")
        
        # Extract and return structured response
        return self._extract_thinking_and_evaluation(raw_response, template_variables)

    def get_available_post_types(self) -> List[str]:
        """Get list of available post types"""
        return list(self.baseline_templates.keys())
    
    def get_available_criteria(self) -> List[str]:
        """Get list of available criteria"""
        return list(self.criteria_templates.keys())
    
    def get_default_criteria(self) -> Dict[str, float]:
        """Get default criteria weights"""
        return self.default_criteria.copy()
    
    def get_post_type_description(self, post_type: str) -> str:
        """Get description of a specific post type"""
        descriptions = {
            "health_nutrition": "Educational posts about food and nutrition with specific data",
            "disease_warning": "Warning posts about diseases and health risks",
            "travel_adventure": "Posts about travel and adventure activities",
            "business_enterprise": "Business-focused posts for enterprises",
            "lifestyle_office": "Posts about office health and lifestyle",
            "holiday_event": "Posts about holidays and special events",
        }
        return descriptions.get(post_type, "Unknown post type")

    def _extract_thinking_and_evaluation(self, raw_response: str, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Extract thinking and evaluation from response"""
        thinking = ""
        
        # Find thinking section
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
        
        # Parse evaluation
        score, feedback = self._parse_evaluation_response(raw_response)
        
        return {
            "thinking": thinking,
            "score": score,
            "feedback": feedback,
            "full_response": raw_response,
            "template_variables": template_vars,
            "language": template_vars.get("language"),
            "post_type": template_vars.get("post_type"),
            "target_audience": template_vars.get("target_audience"),
            "custom_criteria": template_vars.get("custom_criteria"),
            "criteria": template_vars.get("criteria")
        }
    
    def _parse_evaluation_response(self, raw: str) -> tuple[float, str]:
        """Extract JSON result in any format"""
        patterns = [
            r'<result>\s*(\{[^{}]*"score"[^{}]*"feedback"[^{}]*\})\s*</result>',
            r'<evaluation>\s*(\{[^{}]*"score"[^{}]*"feedback"[^{}]*\})\s*</evaluation>',
            r'\{[^{}]*"score"[^{}]*"feedback"[^{}]*\}',
        ]
        
        for pat in patterns:
            m = re.search(pat, raw, re.S | re.I)
            if not m:
                continue
            json_str = m.group(1) if m.groups() else m.group(0)
            try:
                data = json.loads(json_str)
                score = max(0.0, min(1.0, float(data.get("score", 0))))
                feedback = str(data.get("feedback", "No feedback provided"))
                return score, feedback
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"JSON parsing error: {e}")
                continue
        
        # Enhanced fallback parsing
        score_match = re.search(r'score["\']?\s*:\s*([0-9.]+)', raw, re.I)
        feedback_match = re.search(r'feedback["\']?\s*:\s*["\']([^"\']*)["\']', raw, re.I)
        
        if score_match:
            try:
                score = max(0.0, min(1.0, float(score_match.group(1))))
            except:
                score = 0.0
        else:
            score = 0.0
            
        if feedback_match:
            feedback = feedback_match.group(1)
        else:
            feedback = "Cannot parse evaluation response - please check template format"
        
        print(f"Using fallback parsing - Score: {score}, Feedback: {feedback}")
        return score, feedback