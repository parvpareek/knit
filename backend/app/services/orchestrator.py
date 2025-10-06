# backend/app/services/orchestrator.py
from typing import Dict, Any
from .agents import PlannerAgent, RetrieverAgent, QuizGeneratorAgent, ProgressTrackerAgent, ResponseFormatterAgent
import app.core.vectorstore as vectorstore
from app.core.database import db
from langchain.llms import OpenAI

class AgentOrchestrator:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        
        # Initialize agents
        self.retriever = RetrieverAgent(vectorstore)
        self.quiz_generator = QuizGeneratorAgent(self.llm)
        self.progress_tracker = ProgressTrackerAgent(db)
        self.response_formatter = ResponseFormatterAgent(self.llm)
        self.planner = PlannerAgent(
            self.progress_tracker,
            self.retriever,
            self.quiz_generator,
            self.response_formatter
        )
    
    async def process_student_query(self, query: str) -> Dict[str, Any]:
        """Main orchestration method for student queries"""
        
        # Process through planner
        response = await self.planner.execute(query)
        
        if not response.success:
            return {"error": "Processing failed", "details": response.reasoning}
        
        # Log the interaction
        db.log_interaction(
            interaction_type="ask",
            payload={"query": query},
            response=response.data
        )
        
        return response.data
    
    async def generate_quiz(self, topic: str, difficulty: str = None) -> Dict[str, Any]:
        """Generate adaptive quiz for a topic"""
        
        # Get student proficiency for the topic
        proficiency_response = await self.progress_tracker.execute("get", topic)
        
        if proficiency_response.success:
            proficiency_data = proficiency_response.data.get("proficiency", {})
            strength = proficiency_data.get("strength", "new")
            
            # Determine difficulty if not specified
            if not difficulty:
                if strength == "weak":
                    difficulty = "easy"
                elif strength == "improving":
                    difficulty = "medium"
                else:
                    difficulty = "hard"
        
        # Retrieve content for the topic
        retriever_response = await self.retriever.execute(topic, k=5)
        
        if not retriever_response.success:
            return {"error": "Failed to retrieve content"}
        
        context = "\n\n".join(retriever_response.data["documents"])
        
        # Generate quiz
        quiz_response = await self.quiz_generator.execute(
            topic=topic,
            context=context,
            difficulty=difficulty,
            num_questions=3
        )
        
        if not quiz_response.success:
            return {"error": "Failed to generate quiz"}
        
        # Log the interaction
        db.log_interaction(
            interaction_type="quiz_generation",
            payload={"topic": topic, "difficulty": difficulty},
            response=quiz_response.data
        )
        
        return quiz_response.data
    
    async def get_student_progress(self) -> Dict[str, Any]:
        """Get student progress with recommendations"""
        
        # Get all proficiency data
        proficiency_response = await self.progress_tracker.execute("get_all")
        
        if not proficiency_response.success:
            return {"error": "Failed to retrieve progress"}
        
        proficiency_data = proficiency_response.data["proficiency"]
        
        # Generate recommendations
        weak_areas = [topic for topic, data in proficiency_data.items() if data.get("strength") == "weak"]
        strong_areas = [topic for topic, data in proficiency_data.items() if data.get("strength") == "strong"]
        
        if weak_areas:
            recommendation = f"Focus next on {', '.join(weak_areas[:2])}. I'll prepare targeted practice for you."
            summary = f"You're strong in {', '.join(strong_areas[:2]) if strong_areas else 'several areas'} but need more practice in {', '.join(weak_areas[:2])}."
        else:
            recommendation = "Great progress! Ready to tackle more advanced topics?"
            summary = "You're performing well across all topics. Keep up the excellent work!"
        
        return {
            "progress": proficiency_data,
            "agent_recommendation": recommendation,
            "natural_language_summary": summary
        }