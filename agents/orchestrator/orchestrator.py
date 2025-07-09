from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, Template
from utils import call_llm


class OrchestratorAgent:
    def __init__(self, llm, template_dir: str = "agents/orchestrator/templates"):
        self.llm = llm
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
        # Load main template
        self.main_template = self.env.get_template("orchestrator_main.j2")
        
        # Topic template mapping
        self.topic_templates = {
            "food_nutrition": "topics/food_nutrition.j2",
            "disease_warning": "topics/disease_warning.j2", 
            "travel_adventure": "topics/travel_adventure.j2",
            "business_enterprise": "topics/business_enterprise.j2",
            "lifestyle_office": "topics/lifestyle_office.j2",
            "holiday_event": "topics/holiday_event.j2",
        }

    def plan(self, 
             user_request: str,
             language: str = "vietnamese",  # "vietnamese" or "english"
             topic_type: str = "food_nutrition",  # one of the 7 topic types
             target_audience: Optional[str] = None,
             custom_hashtags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate orchestrator plan using Jinja2 templates
        """
        
        # Validate inputs
        if language not in ["vietnamese", "english"]:
            raise ValueError("Language must be 'vietnamese' or 'english'")
            
        if topic_type not in self.topic_templates:
            raise ValueError(f"Topic type must be one of: {list(self.topic_templates.keys())}")
        
        # Load topic-specific template
        topic_template_name = self.topic_templates[topic_type]
        try:
            topic_template = self.env.get_template(topic_template_name)
        except Exception as e:
            raise ValueError(f"Failed to load template {topic_template_name}: {e}")
        
        # Render the main template with all variables
        template_variables = {
            "language": language,
            "topic_type": topic_type,
            "topic_template_name": topic_template_name,
            "target_audience": target_audience,
            "custom_hashtags": custom_hashtags or [],
            "user_request": user_request
        }
        
        # Render topic template first
        topic_content = topic_template.render(**template_variables)
        
        # Add topic content to variables for main template
        template_variables["topic_content"] = topic_content
        
        # Render main template
        prompt = self.main_template.render(**template_variables)
        
        # Get LLM response
        raw_response = call_llm(self.llm, prompt).strip()
        
        # Extract and return structured response
        return self._extract_thinking_and_plan(raw_response, template_variables)

    def get_available_topics(self) -> List[str]:
        """Get list of available topic types"""
        return list(self.topic_templates.keys())
    
    def get_topic_description(self, topic_type: str) -> str:
        """Get description of a specific topic type"""
        descriptions = {
            "food_nutrition": "Educational posts about food and nutrition",
            "disease_warning": "Warning posts about diseases and health risks",
            "travel_adventure": "Posts about travel and adventure activities",
            "business_enterprise": "Business-focused posts for enterprises",
            "lifestyle_office": "Posts about office health and lifestyle",
            "holiday_event": "Posts about holidays and special events",
        }
        return descriptions.get(topic_type, "Unknown topic type")

    @staticmethod
    def _extract_thinking_and_plan(raw_response: str, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Extract thinking and plan from response"""
        thinking = ""
        plan = raw_response

        # Find thinking section
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            # Remove thinking section from plan
            plan = re.sub(r'🧠 CHAIN OF THOUGHT - ORCHESTRATOR:\s*<thinking>.*?</thinking>', '', raw_response, flags=re.DOTALL).strip()

        return {
            "thinking": thinking,
            "plan": plan,
            "full_response": raw_response,
            "template_variables": template_vars,
            "language": template_vars.get("language"),
            "topic_type": template_vars.get("topic_type"),
            "target_audience": template_vars.get("target_audience"),
            "custom_hashtags": template_vars.get("custom_hashtags")
        }