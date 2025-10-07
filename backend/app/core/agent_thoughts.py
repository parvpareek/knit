"""
Agent Thoughts Tracker - Capture and expose AI decision-making in real-time
"""
from typing import List, Dict, Any
from datetime import datetime
import threading

class AgentThoughtsTracker:
    """
    Thread-safe singleton to capture agent thoughts/decisions during a request.
    Thoughts are stored per-request and returned in API responses.
    """
    _local = threading.local()
    
    @classmethod
    def clear(cls):
        """Clear thoughts for current request"""
        cls._local.thoughts = []
    
    @classmethod
    def add(cls, agent: str, thought: str, emoji: str = "🤔", metadata: Dict[str, Any] = None):
        """Add a thought from an agent"""
        if not hasattr(cls._local, 'thoughts'):
            cls._local.thoughts = []
        
        cls._local.thoughts.append({
            "agent": agent,
            "thought": thought,
            "emoji": emoji,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        # Also print to console for debugging
        print(f"[{agent}] {emoji} {thought}")
    
    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        """Get all thoughts for current request"""
        if not hasattr(cls._local, 'thoughts'):
            return []
        return cls._local.thoughts
    
    @classmethod
    def get_summary(cls) -> str:
        """Get a summary of all agent thoughts"""
        thoughts = cls.get_all()
        if not thoughts:
            return "No agent thoughts recorded"
        
        summary_parts = []
        for t in thoughts:
            summary_parts.append(f"{t['emoji']} {t['agent']}: {t['thought']}")
        
        return " → ".join(summary_parts)


# Global instance
thoughts_tracker = AgentThoughtsTracker()

