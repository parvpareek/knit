# backend/app/agents/llm_planner.py
"""
LLM-Based Intelligent Planner Agent
Creates adaptive study plans, evaluates progress, and adapts dynamically
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class PlannerResponse:
    success: bool
    plan: List[Dict[str, Any]]
    reasoning: str
    next_action: str
    evaluation: Optional[Dict] = None

class LLMPlannerAgent:
    """Intelligent LLM-based planner that creates, evaluates, and adapts study plans"""
    
    def __init__(self, llm, database, vectorstore, memory=None):
        self.llm = llm
        self.db = database
        self.vectorstore = vectorstore
        self.memory = memory  # Redis memory for reading segment plans
        self.name = "LLMPlannerAgent"
        
        # Track plan execution state
        self.current_plan = None
        self.completed_steps = []
        self.concept_attempts = {}  # Track how many times we've worked on each concept
        self.max_attempts_per_concept = 3  # Prevent getting stuck
    
    async def create_initial_plan(self, concepts: List[Dict], student_choice: str = "from_beginning", 
                                 document_outline: str = None, learning_roadmap: Dict = None) -> PlannerResponse:
        """Create initial study plan based on concepts, student choice, and document structure"""
        print(f"[{self.name}] Creating initial plan for {len(concepts)} concepts, choice: {student_choice}")
        
        try:
            # Get student profile
            student_profile = self._get_student_profile()
            
            # Analyze concept relationships
            concept_relationships = await self._analyze_concept_relationships(concepts)
            
            # Create intelligent plan using LLM with document context and roadmap
            plan = await self._llm_create_plan(
                concepts=concepts,
                student_choice=student_choice,
                student_profile=student_profile,
                concept_relationships=concept_relationships,
                document_outline=document_outline,
                learning_roadmap=learning_roadmap
            )
            
            self.current_plan = plan
            
            return PlannerResponse(
                success=True,
                plan=plan,
                reasoning=f"Created {len(plan)}-step plan based on student choice: {student_choice}",
                next_action="execute_plan"
            )
            
        except Exception as e:
            return PlannerResponse(
                success=False,
                plan=[],
                reasoning=f"Plan creation failed: {str(e)}",
                next_action="retry"
            )
    
    async def evaluate_and_adapt(self, completed_step: Dict, quiz_results: Dict = None, 
                                student_questions: List[str] = None) -> PlannerResponse:
        """Evaluate progress and adapt plan based on feedback"""
        print(f"[{self.name}] Evaluating progress and adapting plan...")
        
        try:
            # Mark step as completed
            self.completed_steps.append(completed_step)
            
            # Track concept attempts
            topic = completed_step.get("topic", "")
            self.concept_attempts[topic] = self.concept_attempts.get(topic, 0) + 1
            
            # Read session memory context for this topic
            progress = {}
            next_segment = None
            recent_quiz = None
            unclear_segments = []
            if self.memory and topic:
                try:
                    progress = self.memory.get_topic_progress(topic) or {}
                    next_seg = self.memory.get_next_segment(topic)
                    next_segment = next_seg.get("segment_id") if next_seg else None
                    rq = self.memory.get_recent_quiz_results(topic, k=1)
                    if rq:
                        recent_quiz = rq[0]
                        unclear_segments = recent_quiz.get("unclear_segments", [])
                except Exception:
                    pass

            # Get current student profile
            student_profile = self._get_student_profile()
            
            # Evaluate progress using LLM
            evaluation = await self._llm_evaluate_progress(
                completed_steps=self.completed_steps,
                current_plan=self.current_plan,
                quiz_results=quiz_results,
                student_profile=student_profile,
                student_questions=student_questions
            )
            
            # Decide next action based on evaluation
            next_action = self._decide_next_action(evaluation, quiz_results)
            
            # Adapt plan if needed
            if next_action == "adapt_plan":
                # Insert a remediation micro-segment targeting unclear segments if available
                adapted_plan = list(self.current_plan or [])
                if unclear_segments:
                    top_seg = unclear_segments[0]
                    adapted_plan.insert(self._current_index_or_end(), {
                        "step_id": f"remediate_{top_seg}",
                        "action": "study_segment",
                        "topic": topic,
                        "concept_id": completed_step.get("concept_id", ""),
                        "segment_id": top_seg,
                        "segment_title": f"Remediate {top_seg}",
                        "difficulty": "easy",
                        "est_minutes": 5,
                        "why_assigned": "Targeted clarification based on recent quiz"
                    })
                else:
                    adapted_plan = await self._llm_adapt_plan(
                        current_plan=self.current_plan,
                        evaluation=evaluation,
                        quiz_results=quiz_results,
                        completed_steps=self.completed_steps
                    )
                self.current_plan = adapted_plan
                
                return PlannerResponse(
                    success=True,
                    plan=adapted_plan,
                    reasoning=evaluation.get("reasoning", "Plan adapted based on performance"),
                    next_action="continue_plan",
                    evaluation=evaluation
                )
            
            elif next_action == "continue_plan":
                return PlannerResponse(
                    success=True,
                    plan=self.current_plan,
                    reasoning="Progress is good, continuing with current plan",
                    next_action="continue_plan",
                    evaluation=evaluation
                )
            
            elif next_action == "move_forward":
                # Student has mastered this concept, move to next major concept
                # Prefer next segment if available, else move to next concept via LLM
                if next_segment:
                    next_plan = list(self.current_plan or [])
                    next_plan.insert(self._current_index_or_end(), {
                        "step_id": f"study_{next_segment}",
                        "action": "study_segment",
                        "topic": topic,
                        "concept_id": completed_step.get("concept_id", ""),
                        "segment_id": next_segment,
                        "segment_title": f"Next segment {next_segment}",
                        "difficulty": "medium",
                        "est_minutes": 8,
                        "why_assigned": "Advance to next segment in sequence"
                    })
                else:
                    next_plan = await self._llm_create_next_concept_plan(
                        evaluation=evaluation,
                        student_profile=student_profile,
                        previous_concepts=self.completed_steps
                    )
                self.current_plan = next_plan
                
                return PlannerResponse(
                    success=True,
                    plan=next_plan,
                    reasoning="Great progress! Moving to next major concept",
                    next_action="continue_plan",
                    evaluation=evaluation
                )
            
            else:  # clarify_concept
                # Student is struggling, need more clarification
                return PlannerResponse(
                    success=True,
                    plan=self.current_plan,
                    reasoning=evaluation.get("reasoning", "Student needs clarification"),
                    next_action="clarify_concept",
                    evaluation=evaluation
                )
                
        except Exception as e:
            return PlannerResponse(
                success=False,
                plan=self.current_plan or [],
                reasoning=f"Evaluation failed: {str(e)}",
                next_action="retry"
            )
    
    async def _llm_create_plan(self, concepts: List[Dict], student_choice: str, 
                              student_profile: Dict, concept_relationships: Dict,
                              document_outline: str = None, learning_roadmap: Dict = None) -> List[Dict]:
        """Use LLM to create intelligent study plan with document structure awareness"""
        
        # Format concepts with segments for LLM (with Redis memory fallback)
        concepts_str = "\n".join([
            f"- {c.get('label', 'Unknown')} (ID: {c.get('concept_id', 'unknown')})\n  "
            f"Section: {c.get('section_title', 'N/A')}\n  "
            f"Segments: {len(c.get('learning_segments', []))} segments\n  "
            f"Segment Details: " + ", ".join([
                f"{seg.get('segment_id', 'unknown')}: {seg.get('title', 'Untitled')} ({seg.get('estimated_minutes', 0)}min)"
                for seg in c.get('learning_segments', [])
            ])
            for c in concepts
        ])
        
        # If concepts don't have segments, try to get them from Redis memory
        if self.memory:
            for concept in concepts:
                topic = concept.get('label', '')
                if not concept.get('learning_segments') and topic:
                    segments = self.memory.get_segment_plan(topic)
                    if segments:
                        concept['learning_segments'] = segments
                        print(f"[{self.name}] Retrieved {len(segments)} segments from memory for topic: {topic}")
        
        # Re-format concepts string with updated segments
        concepts_str = "\n".join([
            f"- {c.get('label', 'Unknown')} (ID: {c.get('concept_id', 'unknown')})\n  "
            f"Section: {c.get('section_title', 'N/A')}\n  "
            f"Segments: {len(c.get('learning_segments', []))} segments\n  "
            f"Segment Details: " + ", ".join([
                f"{seg.get('segment_id', 'unknown')}: {seg.get('title', 'Untitled')} ({seg.get('estimated_minutes', 0)}min)"
                for seg in c.get('learning_segments', [])
            ])
            for c in concepts
        ])
        
        # Format student profile
        profile_str = self._format_profile_for_llm(student_profile)
        
        # Format relationships
        relationships_str = json.dumps(concept_relationships, indent=2)
        
        # Add document outline and roadmap context if available
        outline_context = ""
        if document_outline and document_outline != "Document structure not available":
            outline_context = f"""
DOCUMENT STRUCTURE (for context):
{document_outline[:500]}...
(This shows how concepts are organized in the source material)
"""
        
        # Add learning roadmap context if available
        roadmap_context = ""
        if learning_roadmap and learning_roadmap.get("chapters"):
            roadmap_context = f"""
LEARNING ROADMAP (for progression planning):
Total Chapters: {learning_roadmap.get('total_chapters', 0)}
Total Sections: {learning_roadmap.get('total_sections', 0)}
Estimated Learning Hours: {learning_roadmap.get('estimated_learning_hours', 0)}

Chapter Progression:
"""
            for i, chapter in enumerate(learning_roadmap.get('chapters', [])[:3]):
                roadmap_context += f"\nChapter {chapter.get('chapter_number', i+1)}: {chapter.get('title', 'Unknown')}"
                roadmap_context += f" (Est. {chapter.get('estimated_hours', 0)} hours)"
                for section in chapter.get('sections', [])[:2]:
                    roadmap_context += f"\n  - {section.get('title', 'Unknown')}"
            
            roadmap_context += "\n\nUse this roadmap to create a logical learning progression that follows the document structure."
        
        prompt = f"""You are an expert tutor creating a personalized study plan.

AVAILABLE CONCEPTS:
{concepts_str}
{outline_context}
{roadmap_context}
STUDENT PROFILE:
{profile_str}

CONCEPT RELATIONSHIPS:
{relationships_str}

STUDENT CHOICE: {student_choice}

Create a study plan that teaches concepts through their segments. Each concept has multiple segments that build on each other.

PLANNING RULES:
1. For "from_beginning": Teach segments in order (seg_1 → seg_2 → seg_3...)
2. Do NOT schedule a quiz after each segment. Schedule ONE topic-level "practice_quiz" AFTER all segments for that topic are completed.
3. For "diagnostic": Use "diagnostic_quiz" to test understanding first
4. For "choose_topics": Start with "calibration_quiz" to assess level
5. Follow segment prerequisites (some segments require others first)
6. Use estimated_minutes from segments for timing

Return a JSON array with this structure:
[
  {{
    "step_id": "step_1",
    "action": "study_segment|practice_quiz|diagnostic_quiz",
    "topic": "Concept name",
    "concept_id": "concept_id",
    "segment_id": "seg_1",
    "segment_title": "Segment title from learning_segments",
    "chapter": "Chapter X: Chapter Title",
    "section": "Section Y: Section Title",
    "difficulty": "easy|medium|hard",
    "est_minutes": 5-10,
    "why_assigned": "Clear explanation of why this segment is needed",
    "prerequisites": ["segment_ids that should be completed first"],
    "roadmap_position": "early|middle|late"
  }}
]

IMPORTANT: Create steps for individual segments, not entire concepts. Each segment should be a separate step.
After finishing all segments for a topic, add ONE topic-level practice_quiz step.
Make the plan logical, achievable, and personalized. Use segment timing estimates.
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            plan = json.loads(response.content)
            
            # Validate and clean plan
            return self._validate_plan(plan, concepts)
            
        except Exception as e:
            # Fallback plan if LLM fails
            return self._create_fallback_plan(concepts, student_choice)
    
    async def _llm_evaluate_progress(self, completed_steps: List[Dict], current_plan: List[Dict],
                                    quiz_results: Dict, student_profile: Dict,
                                    student_questions: List[str]) -> Dict:
        """Use LLM to evaluate student progress"""
        
        completed_str = "\n".join([f"- {s['topic']}: {s.get('action', 'completed')}" for s in completed_steps[-5:]])
        quiz_str = json.dumps(quiz_results, indent=2) if quiz_results else "No quiz results yet"
        profile_str = self._format_profile_for_llm(student_profile)
        questions_str = "\n".join(student_questions[-3:]) if student_questions else "No questions asked"
        
        prompt = f"""Evaluate student's learning progress and recommend next action.

COMPLETED STEPS:
{completed_str}

LATEST QUIZ RESULTS:
{quiz_str}

STUDENT PROFILE:
{profile_str}

RECENT QUESTIONS:
{questions_str}

Analyze:
1. Understanding level: How well has the student grasped the concepts?
2. Struggle areas: What specific topics need more work?
3. Progress rate: Is the student moving too fast or too slow?
4. Concept mastery: Has the student sufficiently mastered current concepts?
5. Readiness: Is the student ready to move to next major concept?

Return JSON:
{{
  "understanding_level": "strong|moderate|weak",
  "mastery_score": 0.0-1.0,
  "struggle_topics": ["topic1", "topic2"],
  "ready_for_next": true|false,
  "needs_clarification": true|false,
  "recommendation": "move_forward|continue_practice|clarify_concept|adapt_plan",
  "reasoning": "Detailed explanation of your assessment",
  "next_focus": "What to focus on next"
}}
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            evaluation = json.loads(response.content)
            return evaluation
            
        except Exception as e:
            # Fallback evaluation
            return {
                "understanding_level": "moderate",
                "mastery_score": 0.5,
                "ready_for_next": False,
                "recommendation": "continue_practice",
                "reasoning": "Need more practice with current concepts"
            }
    
    async def _llm_adapt_plan(self, current_plan: List[Dict], evaluation: Dict,
                             quiz_results: Dict, completed_steps: List[Dict]) -> List[Dict]:
        """Use LLM to adapt the plan based on evaluation"""
        
        plan_str = json.dumps(current_plan, indent=2)
        eval_str = json.dumps(evaluation, indent=2)
        quiz_str = json.dumps(quiz_results, indent=2) if quiz_results else "No quiz results"
        
        prompt = f"""Adapt the study plan based on student performance.

CURRENT PLAN:
{plan_str}

EVALUATION:
{eval_str}

QUIZ RESULTS:
{quiz_str}

Adapt the plan to:
1. Address struggle areas identified
2. Add clarification steps if needed
3. Adjust difficulty level appropriately
4. Add more practice if mastery is low
5. Move forward if student is ready
6. Keep momentum going (don't get stuck on one concept)

Return adapted plan as JSON array with same structure as original plan.
Keep it to 3-5 steps. Focus on immediate next actions.
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            adapted_plan = json.loads(response.content)
            return self._validate_plan(adapted_plan, current_plan)
            
        except Exception as e:
            # Return current plan if adaptation fails
            return current_plan
    
    async def _llm_create_next_concept_plan(self, evaluation: Dict, student_profile: Dict,
                                           previous_concepts: List[Dict]) -> List[Dict]:
        """Create plan for next major concept"""
        
        eval_str = json.dumps(evaluation, indent=2)
        profile_str = self._format_profile_for_llm(student_profile)
        prev_str = "\n".join([f"- {s['topic']}" for s in previous_concepts[-5:]])
        
        prompt = f"""Student has mastered current concepts. Create plan for next major concept.

PREVIOUS CONCEPTS LEARNED:
{prev_str}

EVALUATION:
{eval_str}

STUDENT PROFILE:
{profile_str}

Create a plan that:
1. Introduces the next logical concept in the progression
2. Connects it to previously learned concepts
3. Starts with appropriate difficulty
4. Includes learning + practice steps
5. Maintains engagement and momentum

Return JSON array with 3-5 steps following the standard plan structure.
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            next_plan = json.loads(response.content)
            return next_plan
            
        except Exception as e:
            # Fallback: continue with review
            return self._create_review_plan(previous_concepts)
    
    async def _analyze_concept_relationships(self, concepts: List[Dict]) -> Dict:
        """Analyze how concepts relate to each other"""
        
        # Get sample content for each concept
        relationships = {}
        
        for concept in concepts[:10]:  # Limit to prevent overload
            concept_id = concept['concept_id']
            label = concept['label']
            
            # Query vectorstore for related content
            try:
                results = self.vectorstore.query_top_k(label, k=2)
                documents = results.get('documents', [[]])[0] if 'documents' in results else []
                
                relationships[concept_id] = {
                    "label": label,
                    "prerequisites": self._infer_prerequisites(label, documents),
                    "related_to": self._infer_related_concepts(label, [c['label'] for c in concepts])
                }
            except:
                relationships[concept_id] = {
                    "label": label,
                    "prerequisites": [],
                    "related_to": []
                }
        
        return relationships
    
    def _infer_prerequisites(self, concept: str, documents: List[str]) -> List[str]:
        """Infer prerequisite concepts from content"""
        # Simple heuristic-based prerequisite detection
        prerequisites = []
        
        common_prerequisites = {
            "derivatives": ["limits", "functions"],
            "integration": ["derivatives", "functions"],
            "thermodynamics": ["mechanics", "energy"],
            "optics": ["waves", "light"],
            "electricity": ["charge", "forces"]
        }
        
        concept_lower = concept.lower()
        for key, prereqs in common_prerequisites.items():
            if key in concept_lower:
                prerequisites.extend(prereqs)
        
        return prerequisites
    
    def _infer_related_concepts(self, concept: str, all_concepts: List[str]) -> List[str]:
        """Infer related concepts"""
        related = []
        
        concept_lower = concept.lower()
        for other in all_concepts:
            if other != concept:
                other_lower = other.lower()
                # Simple similarity check
                if any(word in other_lower for word in concept_lower.split()) or \
                   any(word in concept_lower for word in other_lower.split()):
                    related.append(other)
        
        return related[:3]  # Limit to 3 related concepts
    
    def _decide_next_action(self, evaluation: Dict, quiz_results: Dict = None) -> str:
        """Decide next action based on evaluation"""
        # Deterministic rules first
        if quiz_results and isinstance(quiz_results, dict) and "score_percentage" in quiz_results:
            topic = quiz_results.get("topic", "")
            score = quiz_results.get("score_percentage", 0.0) / 100.0 if quiz_results.get("score_percentage") > 1 else quiz_results.get("score_percentage", 0.0)
            # Normalize score to 0..1
            if score < 0.5:
                return "clarify_concept"  # REMEDIATE/REPEAT
            if 0.5 <= score < 0.8:
                return "adapt_plan"  # add remedial micro-step
            if score >= 0.8:
                return "move_forward"

        # Check if stuck on same concept
        if quiz_results:
            topic = quiz_results.get("topic", "")
            attempts = self.concept_attempts.get(topic, 0)
            if attempts >= self.max_attempts_per_concept:
                print(f"[{self.name}] Warning: {attempts} attempts on {topic}, forcing move forward")
                return "move_forward"

        # Fallback to LLM recommendation
        recommendation = evaluation.get("recommendation", "continue_practice")
        if recommendation == "move_forward" and evaluation.get("ready_for_next", False):
            return "move_forward"
        elif recommendation == "clarify_concept" and evaluation.get("needs_clarification", False):
            return "clarify_concept"
        elif recommendation == "adapt_plan":
            return "adapt_plan"
        else:
            return "continue_plan"
    
    def _get_student_profile(self) -> Dict:
        """Get current student profile"""
        try:
            return self.db.get_topic_proficiency()
        except:
            return {}
    
    def _format_profile_for_llm(self, profile: Dict) -> str:
        """Format student profile for LLM"""
        if not profile:
            return "New student, no prior history"
        
        lines = []
        for topic, data in profile.items():
            if isinstance(data, dict):
                accuracy = data.get("accuracy", 0.0)
                attempts = data.get("attempts", 0)
                strength = data.get("strength", "unknown")
                lines.append(f"- {topic}: {accuracy:.1%} accuracy, {attempts} attempts, {strength}")
        
        return "\n".join(lines) if lines else "New student, no prior history"
    
    def _validate_plan(self, plan: List[Dict], reference: Any) -> List[Dict]:
        """Validate and clean plan"""
        validated = []
        
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                continue
            
            # Ensure required fields
            validated_step = {
                "step_id": step.get("step_id", f"step_{i+1}"),
                "action": step.get("action", "study_topic"),
                "topic": step.get("topic", "Unknown Topic"),
                "concept_id": step.get("concept_id", f"concept_{i+1}"),
                "difficulty": step.get("difficulty", "medium"),
                "est_minutes": step.get("est_minutes", 10),
                "why_assigned": step.get("why_assigned", "Practice and reinforcement"),
                "prerequisites": step.get("prerequisites", [])
            }
            
            validated.append(validated_step)
        
        return validated[:5]  # Limit to 5 steps
    
    def _create_fallback_plan(self, concepts: List[Dict], student_choice: str) -> List[Dict]:
        """Create fallback plan if LLM fails"""
        plan = []
        
        if student_choice == "from_beginning":
            # For beginners: alternate study and practice for each concept
            for i, concept in enumerate(concepts[:3]):
                # Study step
                plan.append({
                    "step_id": f"study_{i+1}",
                    "action": "study_topic",
                    "topic": concept.get('label', 'Unknown'),
                    "concept_id": concept.get('concept_id', f'concept_{i}'),
                    "difficulty": "easy",
                    "est_minutes": 10,
                    "why_assigned": f"Learn {concept.get('label', 'this topic')} from basics",
                    "prerequisites": []
                })
                # Practice step  
                plan.append({
                    "step_id": f"practice_{i+1}",
                    "action": "practice_quiz",
                    "topic": concept.get('label', 'Unknown'),
                    "concept_id": concept.get('concept_id', f'concept_{i}'),
                    "difficulty": "easy",
                    "est_minutes": 5,
                    "why_assigned": f"Practice {concept.get('label', 'this topic')} to reinforce learning",
                    "prerequisites": []
                })
        else:
            # For diagnostic/choose_topics: quizzes to assess
            for i, concept in enumerate(concepts[:3]):
                plan.append({
                    "step_id": f"step_{i+1}",
                    "action": "practice_quiz",
                    "topic": concept.get('label', 'Unknown'),
                    "concept_id": concept.get('concept_id', f'concept_{i}'),
                    "difficulty": "medium",
                    "est_minutes": 5,
                    "why_assigned": f"Assess your understanding of {concept.get('label', 'this topic')}",
                    "prerequisites": []
                })
        
        return plan
    
    def _create_review_plan(self, previous_concepts: List[Dict]) -> List[Dict]:
        """Create review plan for previously learned concepts"""
        plan = []
        
        # Review last 2-3 concepts
        for i, concept in enumerate(previous_concepts[-3:]):
            plan.append({
                "step_id": f"review_{i+1}",
                "action": "practice_quiz",
                "topic": concept.get('topic', 'Review'),
                "concept_id": concept.get('concept_id', f'review_{i+1}'),
                "difficulty": "medium",
                "est_minutes": 5,
                "why_assigned": "Review and reinforce previous learning",
                "prerequisites": []
            })
        
        return plan
    
    def reset_state(self):
        """Reset planner state"""
        self.current_plan = None
        self.completed_steps = []
        self.concept_attempts = {}

    def _current_index_or_end(self) -> int:
        try:
            return max(len(self.completed_steps), 0)
        except Exception:
            return 0


