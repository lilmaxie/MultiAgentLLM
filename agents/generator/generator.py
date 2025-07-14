from __future__ import annotations
import re
import json
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, Template
from utils import call_llm


class GeneratorAgent:
    def __init__(self, llm, template_dir: str = "agents/generator/templates"):
        self.llm = llm
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
        # Load main template
        self.main_template = self.env.get_template("generator_main.j2")
        
        # Post type template mapping
        self.post_type_templates = {
            "health_nutrition": "post_types/health_nutrition.j2",
            "disease_warning": "post_types/disease_warning.j2",
            "travel_adventure": "post_types/travel_adventure.j2",
            "business_enterprise": "post_types/business_enterprise.j2",
            "lifestyle_office": "post_types/lifestyle_office.j2",
            "holiday_event": "post_types/holiday_event.j2",
        }

    def _extract_search_content_from_plan(self, plan_data: Any) -> str:
        """Extract search content from orchestrator plan data"""
        search_content = ""
        
        if isinstance(plan_data, dict):
            # Check if search results are available
            search_results = plan_data.get("search_results")
            if search_results and search_results.get("success"):
                search_content = "RESEARCH FINDINGS FROM ORCHESTRATOR:\n\n"
                
                # Add main answer if available
                if search_results.get("answer"):
                    search_content += f"KEY INSIGHTS:\n{search_results['answer']}\n\n"
                
                # Add detailed results
                results = search_results.get("results", [])
                if results:
                    search_content += "DETAILED SOURCES:\n"
                    for i, result in enumerate(results[:5], 1):
                        title = result.get("title", "Unknown")
                        content = result.get("content", "")
                        url = result.get("url", "")
                        score = result.get("score", 0)
                        domain = result.get("domain", "")
                        
                        search_content += f"\n{i}. {title}\n"
                        search_content += f"   Domain: {domain}\n"
                        search_content += f"   URL: {url}\n"
                        search_content += f"   Relevance: {score:.2f}\n"
                        
                        # Add content snippet
                        if content:
                            snippet = content[:300] + "..." if len(content) > 300 else content
                            search_content += f"   Content: {snippet}\n"
        
        return search_content

    def generate(self, 
                 user_request: str,
                 plan_data: Any,
                 language: str = "vietnamese",
                 post_type: str = "health_nutrition",
                 target_audience: Optional[str] = None,
                 custom_hashtags: Optional[List[str]] = None,
                 feedback: str = "") -> Dict[str, Any]:
        """
        Generate content using Jinja2 templates with search content from orchestrator
        """
        
        # Extract plan text from plan_data
        if isinstance(plan_data, dict):
            plan_text = plan_data.get("plan", "")
            # Try to extract other info from plan_data if available
            if not language and "language" in plan_data:
                language = plan_data["language"]
            if not post_type and "topic_type" in plan_data:
                post_type = plan_data["topic_type"]
            if not target_audience and "target_audience" in plan_data:
                target_audience = plan_data["target_audience"]
            if not custom_hashtags and "custom_hashtags" in plan_data:
                custom_hashtags = plan_data["custom_hashtags"]
        else:
            plan_text = str(plan_data)
        
        # Validate inputs
        if language not in ["vietnamese", "english"]:
            raise ValueError("Language must be 'vietnamese' or 'english'")
            
        if post_type not in self.post_type_templates:
            raise ValueError(f"Post type must be one of: {list(self.post_type_templates.keys())}")
        
        # Extract search content from orchestrator plan
        search_content = self._extract_search_content_from_plan(plan_data)
        
        # Load post type specific template
        post_type_template_name = self.post_type_templates[post_type]
        try:
            post_type_template = self.env.get_template(post_type_template_name)
        except Exception as e:
            raise ValueError(f"Failed to load template {post_type_template_name}: {e}")
        
        # Render the post type template with all variables
        template_variables = {
            "language": language,
            "post_type": post_type,
            "post_type_template_name": post_type_template_name,
            "target_audience": target_audience,
            "custom_hashtags": custom_hashtags or [],
            "user_request": user_request,
            "plan_text": plan_text,
            "feedback": feedback,
            "search_content": search_content if search_content else None,
            "search_enabled": bool(search_content)
        }
        
        # Render post type template first
        post_type_content = post_type_template.render(**template_variables)
        
        # Add post type content to variables for main template
        template_variables["post_type_content"] = post_type_content
        
        # Render main template
        prompt = self.main_template.render(**template_variables)
        
        # Get LLM response
        raw_response = call_llm(self.llm, prompt).strip()
        
        # Extract and return structured response
        result = self._extract_thinking_and_content(raw_response, template_variables)
        
        # Add search information to result
        if search_content:
            result["search_content"] = search_content
            result["search_enabled"] = True
        
        return result

    def get_available_post_types(self) -> List[str]:
        """Get list of available post types"""
        return list(self.post_type_templates.keys())
    
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

    def _extract_thinking_and_content(self, raw_response: str, template_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Extract thinking and content from response with strong improvement"""
        thinking = ""
        content = ""
        
        # Find and extract thinking
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', raw_response, re.DOTALL)
        if thinking_match:
            thinking = thinking_match.group(1).strip()
        
        # Find and extract content
        content_match = re.search(r'<content>(.*?)</content>', raw_response, re.DOTALL)
        if content_match:
            content = content_match.group(1).strip()
        else:
            # If no <content> tag, take part after thinking
            content = raw_response
            # Remove all unwanted parts
            unwanted_patterns = [
                r'🧠\s*CHAIN OF THOUGHT.*?(?=\n\n|\n[^🧠]|$)',
                r'<thinking>.*?</thinking>',
                r'<think>.*?</think>',
                r'CONTENT:\s*',
                r'Content as follows:\s*',
                r'Post content:\s*',
                r'Article:\s*',
                r'<content>\s*',
                r'</content>\s*',
                r'---+\s*',
                r'📄\s*FINAL TEMPLATE:\s*',
                r'-{10,}',
                r'^\s*\d+\.\s*\*\*LANGUAGE\*\*.*?(?=\n\n|\n[^*]|$)',
                r'^\s*\d+\.\s*\*\*MAIN TOPIC\*\*.*?(?=\n\n|\n[^*]|$)',
                r'^\s*\d+\.\s*\*\*TARGET AUDIENCE\*\*.*?(?=\n\n|\n[^*]|$)',
                r'^\s*\d+\.\s*\*\*MESSAGE\*\*.*?(?=\n\n|\n[^*]|$)',
                r'^\s*\d+\.\s*\*\*MESSAGE OBJECTIVE\*\*.*?(?=\n\n|\n[^*]|$)',
            ]
            
            for pattern in unwanted_patterns:
                content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        # Remove CONTENT_END at the end (improved regex)
        content = re.sub(r'\s*\*\*CONTENT_END\*\*\s*', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\s*CONTENT_END\s*', '', content, flags=re.IGNORECASE)
        
        # Remove excessive blank lines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()

        return {
            "thinking": thinking,
            "content": content,
            "full_response": raw_response,
            "template_variables": template_vars,
            "language": template_vars.get("language"),
            "post_type": template_vars.get("post_type"),
            "target_audience": template_vars.get("target_audience"),
            "custom_hashtags": template_vars.get("custom_hashtags"),
            "search_enabled": template_vars.get("search_enabled", False)
        }