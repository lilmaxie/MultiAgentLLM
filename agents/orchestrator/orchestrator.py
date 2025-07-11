from __future__ import annotations
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from jinja2 import Environment, FileSystemLoader, Template
from utils import call_llm
import requests
import time


class OrchestratorAgent:
    def __init__(self, llm, tavily_api_key: str, template_dir: str = "agents/orchestrator/templates"):
        self.llm = llm
        self.tavily_api_key = tavily_api_key
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

    def _search_with_tavily(self, query: str) -> Dict[str, Any]:
        """Search using Tavily API with the original user query"""
        try:
            api_url = "https://api.tavily.com/search"
            
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": True,
                "max_results": 5,
                "include_domains": None  # Let Tavily find the best sources
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Process results to extract useful information
            results = data.get("results", [])
            processed_results = []
            
            for result in results:
                processed_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""),
                    "score": result.get("score", 0),
                    "published_date": result.get("published_date", ""),
                    "domain": result.get("url", "").split("/")[2] if result.get("url") else ""
                }
                processed_results.append(processed_result)
            
            return {
                "success": True,
                "answer": data.get("answer", ""),
                "results": processed_results,
                "query": query,
                "total_results": len(processed_results)
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Tavily Search API error: {e}")
            return {"success": False, "error": str(e), "query": query}
        except Exception as e:
            print(f"Unexpected error in Tavily Search: {e}")
            return {"success": False, "error": str(e), "query": query}

    def _format_search_results(self, search_data: Dict[str, Any]) -> str:
        """Format search results for template"""
        if not search_data.get("success"):
            return "No search results available."
        
        formatted_content = f"🔍 SEARCH ANALYSIS:\n"
        formatted_content += f"• Query: {search_data['query']}\n"
        formatted_content += f"• Total sources found: {search_data['total_results']}\n"
        
        # Add main answer if available
        if search_data.get("answer"):
            formatted_content += f"\n📊 KEY INSIGHTS:\n{search_data['answer']}\n"
        
        # Add top results
        results = search_data.get("results", [])
        if results:
            formatted_content += f"\n📋 TOP SOURCES:\n"
            for i, result in enumerate(results[:5], 1):
                title = result.get("title", "Unknown")
                url = result.get("url", "")
                score = result.get("score", 0)
                domain = result.get("domain", "")
                
                formatted_content += f"{i}. {title}\n"
                formatted_content += f"   Domain: {domain}\n"
                formatted_content += f"   URL: {url}\n"
                formatted_content += f"   Relevance: {score:.2f}\n"
                
                # Add content snippet
                content = result.get("content", "")
                if content:
                    snippet = content[:200] + "..." if len(content) > 200 else content
                    formatted_content += f"   Content: {snippet}\n"
                formatted_content += "\n"
        
        return formatted_content

    def plan(self, 
             user_request: str,
             language: str = "vietnamese",
             topic_type: str = "food_nutrition",
             target_audience: Optional[str] = None,
             custom_hashtags: Optional[List[str]] = None,
             enable_search: bool = True) -> Dict[str, Any]:
        """
        Generate orchestrator plan with Tavily search integration
        """
        
        # Validate inputs
        if language not in ["vietnamese", "english"]:
            raise ValueError("Language must be 'vietnamese' or 'english'")
            
        if topic_type not in self.topic_templates:
            raise ValueError(f"Topic type must be one of: {list(self.topic_templates.keys())}")
        
        # Perform Tavily search if enabled
        search_results = None
        if enable_search:
            print("🔍 Performing Tavily search for reliable sources...")
            search_results = self._search_with_tavily(user_request)
        
        # Load topic-specific template
        topic_template_name = self.topic_templates[topic_type]
        try:
            topic_template = self.env.get_template(topic_template_name)
        except Exception as e:
            raise ValueError(f"Failed to load template {topic_template_name}: {e}")
        
        # Prepare template variables
        template_variables = {
            "language": language,
            "topic_type": topic_type,
            "topic_template_name": topic_template_name,
            "target_audience": target_audience,
            "custom_hashtags": custom_hashtags or [],
            "user_request": user_request,
            "search_results": search_results
        }
        
        # Render topic template first
        topic_content = topic_template.render(**template_variables)
        
        # Add topic content to variables for main template
        template_variables["topic_content"] = topic_content
        
        # Add search information to template
        if search_results:
            template_variables["search_summary"] = self._format_search_results(search_results)
        
        # Render main template
        prompt = self.main_template.render(**template_variables)
        
        # Get LLM response
        raw_response = call_llm(self.llm, prompt).strip()
        
        # Extract and return structured response
        result = self._extract_thinking_and_plan(raw_response, template_variables)
        
        # Add search results to the response
        if search_results:
            result["search_results"] = search_results
        
        return result

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
            "custom_hashtags": template_vars.get("custom_hashtags"),
            "search_enabled": template_vars.get("search_results") is not None
        }