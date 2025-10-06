# backend/app/services/agents.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

class AgentType(Enum):
    PLANNER = "planner"
    RETRIEVER = "retriever"
    QUIZ_GENERATOR = "quiz_generator"
    PROGRESS_TRACKER = "progress_tracker"
    RESPONSE_FORMATTER = "response_formatter"

@dataclass
class AgentResponse:
    success: bool
    data: Dict[str, Any]
    reasoning: str
    next_action: Optional[str] = None

class BaseAgent(ABC):
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.name = agent_type.value
    
    @abstractmethod
    async def execute(self, **kwargs) -> AgentResponse:
        pass
    
    def log_action(self, action: str, details: str):
        print(f"[{self.name}] {action}: {details}")

class PlannerAgent(BaseAgent):
    def __init__(self, progress_tracker, retriever, quiz_generator, response_formatter):
        super().__init__(AgentType.PLANNER)
        self.progress_tracker = progress_tracker
        self.retriever = retriever
        self.quiz_generator = quiz_generator
        self.response_formatter = response_formatter
    
    async def execute(self, query: str) -> AgentResponse:
        self.log_action("planning", f"Analyzing query: {query}")
        
        try:
            # Step 1: Analyze query
            query_analysis = self._analyze_query(query)
            topic = query_analysis.get('topic')
            
            # Step 2: Get student context for the topic
            student_context = await self._get_student_context(topic)
            
            # Step 3: Create execution plan
            plan = self._create_plan(query_analysis, student_context)
            
            # Step 4: Execute plan
            execution_results = await self._execute_plan(plan, query, student_context)
            
            return AgentResponse(
                success=True,
                data=execution_results,
                reasoning=f"Successfully executed {len(plan)} step plan"
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Planning failed: {str(e)}"
            )
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the student's query to extract intent and topic"""
        query_lower = query.lower()
        
        # Extract topic (simple keyword matching)
        topics = ['probability', 'algebra', 'calculus', 'statistics', 'geometry', 'trigonometry']
        detected_topic = None
        for topic in topics:
            if topic in query_lower:
                detected_topic = topic
                break
        
        # Determine intent
        if any(word in query_lower for word in ["quiz", "test", "practice", "questions"]):
            intent = "quiz_request"
        elif any(word in query_lower for word in ["explain", "understand", "help", "what is", "how"]):
            intent = "explanation_request"
        elif any(word in query_lower for word in ["progress", "how am i", "my performance", "status"]):
            intent = "progress_request"
        else:
            intent = "general_help"
        
        return {
            "topic": detected_topic,
            "intent": intent,
            "original_query": query
        }
    
    async def _get_student_context(self, topic: str) -> Dict[str, Any]:
        """Get student's proficiency context for the topic"""
        if not topic:
            return {"proficiency": 0.0, "strength": "new", "attempts": 0}
        
        progress_response = await self.progress_tracker.execute(
            action="get", 
            topic=topic
        )
        
        if progress_response.success:
            proficiency_data = progress_response.data.get("proficiency", {})
            return {
                "proficiency": proficiency_data.get("accuracy", 0.0),
                "strength": proficiency_data.get("strength", "new"),
                "attempts": proficiency_data.get("attempts", 0)
            }
        else:
            return {"proficiency": 0.0, "strength": "new", "attempts": 0}
    
    def _create_plan(self, query_analysis: Dict, student_context: Dict) -> List[Dict[str, Any]]:
        """Create execution plan based on query analysis and student context"""
        intent = query_analysis["intent"]
        topic = query_analysis["topic"]
        strength = student_context.get("strength", "new")
        
        plan = []
        
        if intent == "quiz_request":
            # Determine difficulty based on student strength
            if strength == "weak":
                difficulty = "easy"
            elif strength == "improving":
                difficulty = "medium"
            else:
                difficulty = "hard"
            
            plan = [
                {"tool": "retriever", "params": {"query": topic, "k": 5}},
                {"tool": "quiz_generator", "params": {"topic": topic, "difficulty": difficulty, "count": 3}},
                {"tool": "response_formatter", "params": {"format": "quiz_response"}}
            ]
        
        elif intent == "explanation_request":
            plan = [
                {"tool": "retriever", "params": {"query": topic or query_analysis["original_query"], "k": 3}},
                {"tool": "response_formatter", "params": {"format": "explanation_response"}}
            ]
        
        elif intent == "progress_request":
            plan = [
                {"tool": "progress_tracker", "params": {"action": "get_all"}},
                {"tool": "response_formatter", "params": {"format": "progress_response"}}
            ]
        
        else:
            # General help - provide explanation and offer quiz
            plan = [
                {"tool": "retriever", "params": {"query": query_analysis["original_query"], "k": 3}},
                {"tool": "response_formatter", "params": {"format": "general_response"}}
            ]
        
        return plan
    
    async def _execute_plan(self, plan: List[Dict], query: str, student_context: Dict) -> Dict[str, Any]:
        """Execute the plan and collect results"""
        results = {
            "plan_executed": [],
            "retrieved_content": None,
            "quiz": None,
            "progress": None,
            "natural_language_response": ""
        }
        
        for step in plan:
            tool = step["tool"]
            params = step["params"]
            
            if tool == "retriever":
                retriever_response = await self.retriever.execute(**params)
                if retriever_response.success:
                    results["retrieved_content"] = retriever_response.data["documents"]
                    results["sources"] = retriever_response.data["sources"]
                    results["plan_executed"].append("Retrieved relevant content")
            
            elif tool == "quiz_generator":
                # Get context from retriever if available
                context = ""
                if results["retrieved_content"]:
                    context = "\n\n".join(results["retrieved_content"])
                
                quiz_response = await self.quiz_generator.execute(
                    topic=params["topic"],
                    context=context,
                    difficulty=params["difficulty"],
                    num_questions=params["count"]
                )
                if quiz_response.success:
                    results["quiz"] = quiz_response.data
                    results["plan_executed"].append(f"Generated {params['difficulty']} difficulty quiz")
            
            elif tool == "progress_tracker":
                progress_response = await self.progress_tracker.execute(
                    action=params["action"]
                )
                if progress_response.success:
                    results["progress"] = progress_response.data["proficiency"]
                    results["plan_executed"].append("Retrieved student progress")
            
            elif tool == "response_formatter":
                formatter_response = await self.response_formatter.execute(
                    query=query,
                    results=results,
                    student_context=student_context
                )
                if formatter_response.success:
                    results["natural_language_response"] = formatter_response.data["response"]
                    results["plan_executed"].append("Formatted natural language response")
        
        return results

class RetrieverAgent(BaseAgent):
    def __init__(self, vectorstore):
        super().__init__(AgentType.RETRIEVER)
        self.vectorstore = vectorstore
    
    async def execute(self, query: str, k: int = 5) -> AgentResponse:
        self.log_action("retrieving", f"Query: {query}")
        
        try:
            results = self.vectorstore.query_top_k(query, k)
            documents = results.get('documents', [[]])[0] if 'documents' in results else []
            metadatas = results.get('metadatas', [[]])[0] if 'metadatas' in results else []
            
            return AgentResponse(
                success=True,
                data={
                    "documents": documents,
                    "metadatas": metadatas,
                    "sources": [meta.get('source', 'unknown') for meta in metadatas]
                },
                reasoning=f"Retrieved {len(documents)} relevant documents"
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Retrieval failed: {str(e)}"
            )

class QuizGeneratorAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(AgentType.QUIZ_GENERATOR)
        self.llm = llm
    
    async def execute(self, topic: str, context: str, difficulty: str = "medium", num_questions: int = 3) -> AgentResponse:
        self.log_action("generating_quiz", f"Topic: {topic}, Difficulty: {difficulty}")
        
        try:
            quiz = await self._generate_quiz(topic, context, difficulty, num_questions)
            
            return AgentResponse(
                success=True,
                data=quiz,
                reasoning=f"Generated {num_questions} {difficulty} questions for {topic}"
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Quiz generation failed: {str(e)}"
            )
    
    async def _generate_quiz(self, topic: str, context: str, difficulty: str, num_questions: int) -> Dict:
        # Enhanced quiz generation with proper structure
        return {
            "quiz_id": f"qz_{topic}_{difficulty}_{num_questions}",
            "topic": topic,
            "difficulty": difficulty,
            "why_assigned": f"Generated {difficulty} difficulty quiz to reinforce {topic} concepts",
            "questions": [
                {
                    "question_id": f"q{i+1}",
                    "text": f"Sample {difficulty} question {i+1} about {topic}",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "explanation": f"Explanation for {difficulty} question {i+1}",
                    "hint": f"Hint for {difficulty} question {i+1}"
                }
                for i in range(num_questions)
            ]
        }

class ProgressTrackerAgent(BaseAgent):
    def __init__(self, database):
        super().__init__(AgentType.PROGRESS_TRACKER)
        self.db = database
    
    async def execute(self, action: str = "get", topic: str = None, accuracy: float = None) -> AgentResponse:
        self.log_action("tracking_progress", f"Action: {action}")
        
        try:
            if action == "get":
                proficiency = self.db.get_topic_proficiency(topic)
                return AgentResponse(
                    success=True,
                    data={"proficiency": proficiency},
                    reasoning=f"Retrieved progress for topic {topic or 'all topics'}"
                )
            elif action == "get_all":
                proficiency = self.db.get_topic_proficiency()
                return AgentResponse(
                    success=True,
                    data={"proficiency": proficiency},
                    reasoning="Retrieved all progress data"
                )
            elif action == "update":
                success = self.db.update_topic_proficiency(topic, accuracy)
                return AgentResponse(
                    success=success,
                    data={"updated": success},
                    reasoning=f"Updated progress for {topic} with accuracy {accuracy}"
                )
            else:
                return AgentResponse(
                    success=False,
                    data={},
                    reasoning=f"Unknown action: {action}"
                )
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Progress tracking failed: {str(e)}"
            )

class ResponseFormatterAgent(BaseAgent):
    def __init__(self, llm):
        super().__init__(AgentType.RESPONSE_FORMATTER)
        self.llm = llm
    
    async def execute(self, query: str, results: Dict, student_context: Dict) -> AgentResponse:
        self.log_action("formatting_response", f"Formatting response for query: {query}")
        
        try:
            response = self._format_response(query, results, student_context)
            
            return AgentResponse(
                success=True,
                data={"response": response},
                reasoning="Formatted natural language response"
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                reasoning=f"Response formatting failed: {str(e)}"
            )
    
    def _format_response(self, query: str, results: Dict, student_context: Dict) -> str:
        """Format response based on available results"""
        
        if results.get("quiz"):
            quiz = results["quiz"]
            return f"I've prepared a {quiz['difficulty']} difficulty quiz on {quiz['topic']}. {quiz['why_assigned']} Let's work through these questions together!"
        
        elif results.get("retrieved_content"):
            content = results["retrieved_content"][0] if results["retrieved_content"] else ""
            return f"Based on your question about '{query}', here's what I found: {content[:200]}... Would you like me to explain this further or create some practice questions?"
        
        elif results.get("progress"):
            progress = results["progress"]
            weak_areas = [topic for topic, data in progress.items() if data.get("strength") == "weak"]
            if weak_areas:
                return f"I can see you're working on several topics. Your weak areas are: {', '.join(weak_areas[:2])}. Let me help you strengthen these areas with targeted practice."
            else:
                return "Great job! You're making good progress. Is there any specific topic you'd like to focus on today?"
        
        else:
            return "I'm here to help you learn! Could you tell me more about what you'd like to work on today?"