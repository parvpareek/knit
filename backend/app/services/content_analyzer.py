# backend/app/services/content_analyzer.py
"""
Content Analysis Service for Adaptive AI Tutor
Analyzes ingested documents to extract key concepts, skills, and learning objectives.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from app.core.config import settings
import json

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    temperature=0.1,  # Lower temperature for more consistent analysis
    google_api_key=settings.GOOGLE_API_KEY
)

class LearningObjective(BaseModel):
    """Schema for a learning objective"""
    concept: str = Field(description="The main concept or topic")
    skill: str = Field(description="The specific skill to be mastered")
    difficulty_level: str = Field(description="beginner/intermediate/advanced")
    prerequisites: List[str] = Field(description="Required prior knowledge")
    exam_relevance: str = Field(description="How important this is for the target exam")
    practice_questions_needed: int = Field(description="Number of practice questions needed")

class ExamSyllabus(BaseModel):
    """Schema for exam syllabus analysis"""
    exam_name: str
    total_concepts: int
    learning_objectives: List[LearningObjective]
    difficulty_distribution: Dict[str, int]  # {"beginner": 10, "intermediate": 15, "advanced": 5}
    estimated_study_hours: int
    key_topics: List[str]

def analyze_document_content(document_text: str, target_exam: str = "JEE") -> Dict[str, Any]:
    """
    Analyze document content to extract learning objectives and key concepts.
    """
    parser = JsonOutputParser(pydantic_object=ExamSyllabus)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert educational content analyzer for {target_exam} preparation.
        
Your task is to analyze study material and extract:
1. Key concepts and learning objectives
2. Skill requirements for each concept
3. Difficulty levels and prerequisites
4. Exam relevance and importance
5. Practice question requirements

Focus on:
- Mathematical concepts (algebra, calculus, probability, etc.)
- Problem-solving skills
- Conceptual understanding vs. memorization
- Exam-specific requirements

Be thorough but concise. Prioritize concepts that are commonly tested."""),
        
        ("user", """Analyze this study material and extract learning objectives:

Document Content:
{document_text}

Target Exam: {target_exam}

{format_instructions}

Return a comprehensive analysis focusing on actionable learning objectives.""")
    ])
    
    try:
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "document_text": document_text[:4000],  # Limit to avoid token limits
            "target_exam": target_exam,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    except Exception as e:
        # Fallback analysis
        return {
            "exam_name": target_exam,
            "total_concepts": 5,
            "learning_objectives": [
                {
                    "concept": "Basic Probability",
                    "skill": "Calculate probability of events",
                    "difficulty_level": "beginner",
                    "prerequisites": ["Basic arithmetic"],
                    "exam_relevance": "high",
                    "practice_questions_needed": 10
                }
            ],
            "difficulty_distribution": {"beginner": 3, "intermediate": 2, "advanced": 0},
            "estimated_study_hours": 8,
            "key_topics": ["probability", "statistics"],
            "error": str(e)
        }

def identify_student_gaps(student_proficiency: Dict[str, Any], 
                         learning_objectives: List[Dict]) -> List[Dict[str, Any]]:
    """
    Identify gaps between student's current proficiency and required learning objectives.
    """
    gaps = []
    
    for objective in learning_objectives:
        concept = objective.get("concept", "")
        difficulty = objective.get("difficulty_level", "intermediate")
        
        # Get student's proficiency for this concept
        # Handle both dict and float formats
        concept_key = concept.lower().replace(" ", "_")
        if concept_key in student_proficiency:
            if isinstance(student_proficiency[concept_key], dict):
                student_score = student_proficiency[concept_key].get("accuracy", 0.0)
            else:
                student_score = float(student_proficiency[concept_key])
        else:
            student_score = 0.0
        
        # Determine if this is a gap
        required_score = 0.6 if difficulty == "beginner" else 0.7 if difficulty == "intermediate" else 0.8
        
        if student_score < required_score:
            gap_priority = "high" if objective.get("exam_relevance") == "high" else "medium"
            
            gaps.append({
                "concept": concept,
                "current_proficiency": student_score,
                "required_proficiency": required_score,
                "gap_size": required_score - student_score,
                "priority": gap_priority,
                "difficulty_level": difficulty,
                "skill": objective.get("skill", ""),
                "practice_questions_needed": objective.get("practice_questions_needed", 5)
            })
    
    # Sort by priority and gap size
    gaps.sort(key=lambda x: (x["priority"] == "high", -x["gap_size"]), reverse=True)
    
    return gaps

def create_adaptive_quiz_plan(student_gaps: List[Dict], 
                             available_time: int = 30) -> Dict[str, Any]:
    """
    Create an adaptive quiz plan based on student gaps and available time.
    """
    if not student_gaps:
        return {
            "quiz_plan": [],
            "total_questions": 0,
            "estimated_time": 0,
            "focus_areas": []
        }
    
    quiz_plan = []
    total_questions = 0
    focus_areas = []
    
    # Prioritize high-priority gaps
    high_priority_gaps = [gap for gap in student_gaps if gap["priority"] == "high"]
    medium_priority_gaps = [gap for gap in student_gaps if gap["priority"] == "medium"]
    
    # Allocate questions based on priority and gap size
    for gap in high_priority_gaps[:3]:  # Focus on top 3 high-priority gaps
        questions_needed = min(gap["practice_questions_needed"], 8)
        quiz_plan.append({
            "concept": gap["concept"],
            "difficulty": gap["difficulty_level"],
            "questions": questions_needed,
            "skill": gap["skill"],
            "priority": "high"
        })
        total_questions += questions_needed
        focus_areas.append(gap["concept"])
    
    # Add medium priority gaps if time allows
    remaining_time = available_time - (total_questions * 2)  # 2 minutes per question
    for gap in medium_priority_gaps[:2]:
        if remaining_time > 0:
            questions_needed = min(gap["practice_questions_needed"], 5)
            quiz_plan.append({
                "concept": gap["concept"],
                "difficulty": gap["difficulty_level"],
                "questions": questions_needed,
                "skill": gap["skill"],
                "priority": "medium"
            })
            total_questions += questions_needed
            focus_areas.append(gap["concept"])
            remaining_time -= questions_needed * 2
    
    return {
        "quiz_plan": quiz_plan,
        "total_questions": total_questions,
        "estimated_time": total_questions * 2,
        "focus_areas": focus_areas,
        "high_priority_concepts": [gap["concept"] for gap in high_priority_gaps[:3]]
    }

def generate_concept_quiz(concept: str, difficulty: str, skill: str, num_questions: int = 5) -> Dict[str, Any]:
    """
    Generate a targeted quiz for a specific concept and skill.
    """
    parser = JsonOutputParser(pydantic_object=dict)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert quiz generator for {difficulty} level {concept} questions.
        
Focus on testing the specific skill: {skill}

Create questions that:
1. Test conceptual understanding, not just memorization
2. Are appropriate for {difficulty} level
3. Require problem-solving and reasoning
4. Have clear, unambiguous correct answers
5. Include common misconceptions as wrong options

Each question should have exactly 4 options (A, B, C, D) and test the specific skill mentioned."""),
        
        ("user", """Generate {num_questions} {difficulty} level questions about {concept}.

Focus on testing: {skill}

Make sure questions are:
- Clear and unambiguous
- Test conceptual understanding
- Include realistic wrong options
- Appropriate for exam preparation

{format_instructions}""")
    ])
    
    try:
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "concept": concept,
            "difficulty": difficulty,
            "skill": skill,
            "num_questions": num_questions,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    except Exception as e:
        # Fallback quiz generation
        return {
            "quiz_id": f"quiz_{concept}_{difficulty}",
            "concept": concept,
            "difficulty": difficulty,
            "skill": skill,
            "questions": [
                {
                    "question_id": f"q{i+1}",
                    "question_text": f"Sample {difficulty} question about {concept} - {skill}",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct_answer": "A",
                    "explanation": f"This tests your understanding of {skill} in {concept}",
                    "difficulty": difficulty,
                    "topic": concept
                }
                for i in range(num_questions)
            ],
            "error": str(e)
        }
