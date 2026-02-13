"""
Contextual and Reasoning Expert Agent - IMPLEMENTED (Gemini API)

INPUT TYPES:
- Natural language queries (text)
- Context from all other expert outputs

FUNCTIONALITY:
- Processes natural language queries from operators
- Aggregates and interprets context from all other experts
- Provides intelligent insights and recommendations
- Performs risk assessment based on all available data
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNING: google-generativeai not installed. Run: pip install google-generativeai")


class QueryType(Enum):
    """Types of queries the reasoning expert can handle."""
    STATUS_INQUIRY = "status_inquiry"
    RISK_ASSESSMENT = "risk_assessment"
    RECOMMENDATION = "recommendation"
    ANOMALY_EXPLANATION = "anomaly_explanation"
    HISTORICAL_COMPARISON = "historical_comparison"
    PREDICTION = "prediction"
    GENERAL = "general"


@dataclass
class ReasoningResult:
    """Result from a single query processing."""
    query: str
    query_type: QueryType
    response: str = ""
    confidence: float = 0.0
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "pending"


# System prompt for Gemini
SYSTEM_PROMPT = """You are an expert AI assistant for railway track safety monitoring and tampering detection.

Your role is to:
1. Analyze data from multiple sensor systems (geometric sensors, accelerometers, DAS, CCTV, drones, LiDAR, thermal cameras)
2. Interpret findings from specialized analysis modules
3. Provide clear, actionable insights about track safety
4. Assess risk levels and recommend appropriate actions
5. Answer operator questions about track conditions

You have access to analysis results from:
- Track Structural Analysis Expert: Analyzes CSV data (geometry, vibration, acoustic)
- Visual Track Integrity Expert: Analyzes images/videos (defects, obstacles, tampering)
- Thermal Anomaly Interpretation Expert: Analyzes LiDAR/thermal data (3D structure, temperature)

IMPORTANT: Some experts may show "PLACEHOLDER" status - this means the analysis is not yet implemented.
Focus on interpreting data from experts that have actual results.

Be concise, technical when needed, and always prioritize safety.

When assessing risks, consider:
- Immediate safety threats (obstacles, tampering, structural failures)
- Degradation patterns that could lead to future problems
- Environmental factors (weather, temperature variations)
- Historical context when available

Format your responses with clear sections:
1. **Summary**: Brief overview
2. **Findings**: Key observations from expert data
3. **Risk Assessment**: Current risk level and factors
4. **Recommendations**: Actionable next steps

If "Historical Analysis Context" is provided:
- You can answer questions about past trends, total alerts, or recent incidents.
- Use the provided history list to calculate counts or summarize events.
"""


def _classify_query(query: str) -> QueryType:
    """Classify the type of query."""
    query_lower = query.lower()
    
    if any(w in query_lower for w in ["risk", "danger", "threat", "hazard", "safe"]):
        return QueryType.RISK_ASSESSMENT
    if any(w in query_lower for w in ["should", "recommend", "suggest", "what to do", "action"]):
        return QueryType.RECOMMENDATION
    if any(w in query_lower for w in ["status", "current", "now", "condition", "state"]):
        return QueryType.STATUS_INQUIRY
    if any(w in query_lower for w in ["explain", "why", "what caused", "reason"]):
        return QueryType.ANOMALY_EXPLANATION
    if any(w in query_lower for w in ["compare", "history", "previous", "trend", "change"]):
        return QueryType.HISTORICAL_COMPARISON
    if any(w in query_lower for w in ["predict", "forecast", "future", "will", "expect"]):
        return QueryType.PREDICTION
    
    return QueryType.GENERAL


def _build_context_string(context: Dict[str, Any]) -> str:
    """Build a context string from expert results."""
    context_parts = []
    
    # Track Structural Analysis
    if context.get("track_structural"):
        ts = context["track_structural"]
        context_parts.append(f"""
## Track Structural Analysis Expert Results
- Status: {ts.get('status', 'N/A')}
- Confidence: {ts.get('confidence', 0):.1%}
- Alerts: {', '.join(ts.get('alerts', [])) or 'None'}
- Output: {json.dumps(ts.get('output', {}), indent=2, default=str)}
""")
    
    # Visual Integrity
    if context.get("visual_integrity"):
        vi = context["visual_integrity"]
        context_parts.append(f"""
## Visual Track Integrity Expert Results
- Status: {vi.get('status', 'N/A')}
- Confidence: {vi.get('confidence', 0):.1%}
- Alerts: {', '.join(vi.get('alerts', [])) or 'None'}
- Output: {json.dumps(vi.get('output', {}), indent=2, default=str)}
""")
    
    if context.get("thermal_anomaly"):
        ta = context["thermal_anomaly"]
        context_parts.append(f"""
## Thermal Anomaly Interpretation Expert Results
- Status: {ta.get('status', 'N/A')}
- Confidence: {ta.get('confidence', 0):.1%}
- Alerts: {', '.join(ta.get('alerts', [])) or 'None'}
- Output: {json.dumps(ta.get('output', {}), indent=2, default=str)}
""")

    # Historical Context
    if context.get("history"):
        hist = context["history"]
        context_parts.append(f"""
## Historical Analysis Context (Last {len(hist)} records)
{json.dumps(hist, indent=2, default=str)}
""")
    
    if not context_parts:
        return "No analysis context available from other experts."
    
    return "\n".join(context_parts)


def _extract_insights(response: str) -> List[str]:
    """Extract key insights from response."""
    insights = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith(('- ', '• ', '* ', '→ ')):
            insights.append(line.lstrip('-•*→ '))
    
    return insights[:5]


def _extract_recommendations(response: str) -> List[str]:
    """Extract recommendations from response."""
    recommendations = []
    lines = response.split('\n')
    in_rec_section = False
    
    for line in lines:
        line_lower = line.lower()
        if 'recommendation' in line_lower or 'action' in line_lower:
            in_rec_section = True
            continue
        
        if in_rec_section:
            line = line.strip()
            if line.startswith(('- ', '• ', '* ', '1', '2', '3', '4', '5')):
                rec = line.lstrip('-•*0123456789. ')
                if rec:
                    recommendations.append(rec)
            elif line.startswith('#'):
                in_rec_section = False
    
    return recommendations[:5]


def process_with_gemini(queries: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process queries using Gemini API.
    
    Args:
        queries: List of natural language queries
        context: Dictionary with results from other experts
    
    Returns:
        Dictionary with processed results
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    results = {
        "queries_processed": [],
        "insights": [],
        "recommendations": [],
        "alerts": [],
        "confidence": 0.0,
        "status": "pending"
    }
    
    # Check if Gemini is available
    if not GEMINI_AVAILABLE:
        results["status"] = "error"
        results["alerts"].append("Gemini library not installed. Run: pip install google-generativeai")
        return results
    
    # Check API key
    if not api_key or api_key == "your_gemini_api_key_here":
        results["status"] = "error"
        results["alerts"].append("GEMINI_API_KEY not configured. Add your API key to .env file.")
        return results
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Build context string
        context_str = _build_context_string(context)
        
        # Process each query
        all_insights = []
        all_recommendations = []
        
        for query in queries:
            query_type = _classify_query(query)
            
            # Build prompt
            prompt = f"""{SYSTEM_PROMPT}

## Current Analysis Context
{context_str}

## Operator Query
{query}

Please provide a comprehensive response addressing the query based on the analysis context provided.
"""
            
            # Call Gemini
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Extract structured data
            insights = _extract_insights(response_text)
            recommendations = _extract_recommendations(response_text)
            
            all_insights.extend(insights)
            all_recommendations.extend(recommendations)
            
            results["queries_processed"].append({
                "query": query,
                "query_type": query_type.value,
                "response": response_text,
                "insights": insights,
                "recommendations": recommendations
            })
        
        # Aggregate results
        results["insights"] = list(dict.fromkeys(all_insights))[:10]
        results["recommendations"] = list(dict.fromkeys(all_recommendations))[:10]
        results["confidence"] = 0.85
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "error"
        results["alerts"].append(f"Gemini API error: {str(e)}")
    
    return results


def query_gemini(
    question: str,
    track_structural_result: Dict[str, Any] = None,
    visual_integrity_result: Dict[str, Any] = None,
    thermal_anomaly_result: Dict[str, Any] = None,
    history_context: List[Dict[str, Any]] = None
) -> ReasoningResult:
    """
    Convenience function to query Gemini with a single question.
    
    Args:
        question: Natural language question
        track_structural_result: Result from track structural expert
        visual_integrity_result: Result from visual integrity expert
        thermal_anomaly_result: Result from thermal anomaly expert
        history_context: List of recent historical analysis records
    
    Returns:
        ReasoningResult with response and insights
    """
    context = {
        "track_structural": track_structural_result,
        "visual_integrity": visual_integrity_result,
        "thermal_anomaly": thermal_anomaly_result,
        "history": history_context
    }
    
    result = process_with_gemini([question], context)
    
    if result["queries_processed"]:
        processed = result["queries_processed"][0]
        return ReasoningResult(
            query=question,
            query_type=QueryType(processed["query_type"]),
            response=processed["response"],
            confidence=result["confidence"],
            insights=processed["insights"],
            recommendations=processed["recommendations"],
            status=result["status"]
        )
    
    return ReasoningResult(
        query=question,
        query_type=_classify_query(question),
        response="Failed to process query",
        status="error"
    )


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Contextual Reasoning Expert - Gemini Integration Test")
    print("=" * 60)
    
    # Check configuration
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("\n  GEMINI_API_KEY not configured!")
        print("   1. Get API key from: https://makersuite.google.com/app/apikey")
        print("   2. Add to .env file: GEMINI_API_KEY=your_key_here")
    else:
        print("\n GEMINI_API_KEY configured")
        
        # Test query
        print("\nTesting query...")
        result = query_gemini(
            "What is the current safety status of the track?",
            track_structural_result={
                "status": "success",
                "confidence": 0.0,
                "alerts": ["PLACEHOLDER: Track structural analysis not yet implemented"],
                "output": [{"status": "placeholder_analysis"}]
            }
        )
        print(f"\nQuery: {result.query}")
        print(f"Status: {result.status}")
        print(f"Response: {result.response[:500]}...")
