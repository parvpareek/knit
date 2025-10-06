# backend/app/agents/workflow_orchestrator.py
"""
Workflow Orchestrator for Simplified 4-Agent System
Manages the complete student learning workflow
"""

from typing import Dict, List, Any, Optional
from .simple_workflow import (
    SimpleState, IngestAgent, ConceptExtractionAgent, 
    TutorEvaluatorAgent, SpacedRepetitionScheduler,
    AgentType
)
from .llm_planner import LLMPlannerAgent
from app.core.vectorstore import vectorstore
from app.core.database import db
from app.core.config import settings
from app.core.session_memory import SessionMemory, get_redis_client
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from app.utils.session_logger import init_session_logger, get_logger
import asyncio
import time
from app.graphs.adaptive_graph import create_adaptive_graph
from app.graphs.adaptive_state import AdaptiveState, create_initial_state

class WorkflowOrchestrator:
    """Main orchestrator for the simplified 4-agent workflow"""
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.2,
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        # Initialize agents
        self.ingest_agent = IngestAgent(vectorstore)
        self.concept_agent = ConceptExtractionAgent(vectorstore, self.llm)
        self.planner_agent = LLMPlannerAgent(self.llm, db, vectorstore)
        self.tutor_evaluator = TutorEvaluatorAgent(vectorstore, self.llm, db)
        
        # Initialize spaced repetition
        self.spaced_rep = SpacedRepetitionScheduler()
        
        # Initialize Redis memory (with fallback to in-memory)
        self.redis_client = get_redis_client()
        self.memory = None  # Will be initialized per session
        
        # Current state
        self.current_state = SimpleState()
        self.current_state.taught_content = {}  # Initialize taught content storage
        
        # Adaptive graph state (optional, for graph-based routing)
        self.adaptive_state: Optional[AdaptiveState] = None
        self.adaptive_graph = None
        self.use_adaptive_graph = True  # Feature flag - ENABLED

    def _extract_full_text_from_possible_json(self, raw: str) -> str:
        """If raw looks like JSON (possibly fenced), extract full_text; otherwise return raw."""
        try:
            content_raw = (raw or "").strip()
            # Prefer fenced block capture
            if '```json' in content_raw:
                start = content_raw.find('```json') + len('```json')
                end = content_raw.find('```', start)
                if end != -1:
                    content_raw = content_raw[start:end].strip()
            # Try object parse
            import json as _json
            import re as _re
            obj_txt = content_raw
            if not obj_txt.strip().startswith('{'):
                m = _re.search(r"\{[\s\S]*\}", content_raw)
                if m:
                    obj_txt = m.group(0)
            obj = _json.loads(obj_txt)
            if isinstance(obj, dict) and obj.get("full_text"):
                return obj.get("full_text", "")
        except Exception:
            pass
        return raw
    
    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate multiple search queries for better retrieval coverage.
        Uses simple heuristics - no LLM call needed.
        """
        queries = []
        
        # Main query
        queries.append(topic)
        
        # Add explanatory query
        queries.append(f"What is {topic} and how does it work")
        
        # Add technical query
        queries.append(f"{topic} definition explanation examples")
        
        # Add components query for complex topics
        if any(word in topic.lower() for word in ['architecture', 'model', 'system', 'framework']):
            queries.append(f"{topic} components structure")
        
        # Add mechanism query for processes
        if any(word in topic.lower() for word in ['attention', 'mechanism', 'process', 'algorithm']):
            queries.append(f"{topic} mechanism mathematical formulation")
        
        return queries[:4]  # Limit to 4 queries max
    
    def _init_session_memory(self, session_id: str, metadata: Dict, clear_existing: bool = True) -> bool:
        """
        Initialize Redis memory for a session.
        
        Args:
            session_id: Unique session identifier
            metadata: Session metadata to store
            clear_existing: If True, clear any existing data for this session (default: True for new uploads)
        """
        try:
            if self.redis_client:
                # Create new SessionMemory instance for this session
                self.memory = SessionMemory(self.redis_client, session_id)
                
                # Initialize with automatic cleanup of any stale data
                success = self.memory.init_session(metadata, clear_existing=clear_existing)
                if success:
                    print(f"✅ Redis memory initialized for session: {session_id} (cleared={clear_existing})")
                else:
                    print(f"⚠️ Redis memory initialization failed, using in-memory fallback")
                    self.memory = None
                return success
            else:
                print(f"⚠️ Redis not available, using in-memory fallback")
                self.memory = None
                return False
        except Exception as e:
            print(f"⚠️ Memory initialization error: {e}, using in-memory fallback")
            self.memory = None
            return False
    
    async def start_learning_session(self, file_content, filename: str, 
                                   student_choice: str = "diagnostic") -> Dict[str, Any]:
        """Start a complete learning session"""
        print("🚀 Starting new learning session...")
        print(f"[WORKFLOW] File: {filename}, Choice: {student_choice}, Content length: {len(file_content)}")
        
        try:
            # Enforce: clear cross-session summaries only (preserve document index)
            try:
                from app.core.vectorstore import clear_taught_summaries
                clear_taught_summaries()
                print("[WORKFLOW] Cleared taught summaries for new upload")
            except Exception as e:
                print(f"[WORKFLOW] Warning: failed to clear taught summaries on upload: {e}")
            # Step 1: Ingest document
            print("📚 Step 1: Ingesting document...")
            # Log agent action
            try:
                get_logger().log_agent_action("WorkflowOrchestrator", "start_session", f"filename={filename}, choice={student_choice}")
            except Exception:
                pass
            ingest_result = await self.ingest_agent.execute(file_content, filename)
            
            if not ingest_result.success:
                return {
                    "success": False,
                    "error": "Document ingestion failed",
                    "details": ingest_result.reasoning
                }
            
            self.current_state.doc_id = ingest_result.data["doc_id"]
            self.current_state.ingest_status = ingest_result.data["ingest_status"]
            document_outline = ingest_result.data.get("document_outline", "")
            learning_roadmap = ingest_result.data.get("learning_roadmap", {})
            hierarchy = ingest_result.data.get("hierarchy", {})
            
            # Initialize session with a unique session_id per run (avoid collisions across same doc)
            try:
                import uuid
                session_id = f"session_{self.current_state.doc_id}_{uuid.uuid4().hex[:8]}"
                self.current_state.session_id = session_id
                init_session_logger(session_id)
                logger = get_logger()
                logger.log_agent_action("IngestAgent", "parse_document", f"doc_id={self.current_state.doc_id}")
                # Log complete structures
                logger.log_pdf_structure(hierarchy)
                logger.log_data("document_outline", document_outline)
                logger.log_data("learning_roadmap", learning_roadmap)
            except Exception:
                pass
            
            # Initialize Redis memory for this session
            session_metadata = {
                "filename": filename,
                "student_choice": student_choice,
                "doc_id": self.current_state.doc_id,
                "total_chapters": learning_roadmap.get('total_chapters', 0),
                "total_sections": learning_roadmap.get('total_sections', 0)
            }
            self._init_session_memory(session_id, session_metadata)
            
            # Update agents with memory reference
            self.concept_agent.memory = self.memory
            self.planner_agent.memory = self.memory

            # Log parsed structure
            print(f"\n{'='*80}")
            print(f"📄 PARSED DOCUMENT STRUCTURE")
            print(f"{'='*80}")
            print(f"Document ID: {self.current_state.doc_id}")
            print(f"Roadmap: {learning_roadmap.get('total_chapters', 0)} chapters, {learning_roadmap.get('total_sections', 0)} sections")
            print(f"Outline (first 500 chars):\n{document_outline[:500]}...")
            print(f"{'='*80}\n")
            
            # Step 2: Extract concepts using deep document structure
            print("🔍 Step 2: Extracting concepts from deep document structure...")
            try:
                get_logger().log_agent_action("ConceptExtractionAgent", "extract_concepts", "using outline + roadmap")
            except Exception:
                pass
            concept_result = await self.concept_agent.execute(
                self.current_state.doc_id, 
                document_outline=document_outline,
                learning_roadmap=learning_roadmap,
                target_exam="JEE",  # Could be parameterized
                student_context=None  # Could add student context here
            )
            
            if not concept_result.success:
                return {
                    "success": False,
                    "error": "Concept extraction failed",
                    "details": concept_result.reasoning
                }
            
            self.current_state.concepts = concept_result.data["concepts"]
            self.current_state.learning_roadmap = concept_result.data.get("learning_roadmap", {})
            
            # Step 3: Create study plan using LLM planner with deep structure context
            print("📋 Step 3: Creating intelligent study plan with roadmap...")
            try:
                get_logger().log_agent_action("LLMPlannerAgent", "create_initial_plan", f"concepts={len(self.current_state.concepts)}")
            except Exception:
                pass
            plan_result = await self.planner_agent.create_initial_plan(
                self.current_state.concepts, 
                student_choice,
                document_outline=document_outline,
                learning_roadmap=learning_roadmap  # Pass roadmap to planner
            )
            
            if not plan_result.success:
                return {
                    "success": False,
                    "error": "Study planning failed",
                    "details": plan_result.reasoning
                }
            
            self.current_state.study_plan = plan_result.plan
            self.current_state.student_choice = student_choice
            
            # Create lesson session for history tracking
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            
            # Get exam context from Redis if set
            exam_context_data = None
            if self.memory:
                try:
                    exam_context_data = self.memory.get_exam_context()
                    if exam_context_data and exam_context_data.get("exam_type") == "general":
                        exam_context_data = None  # Don't store default
                except Exception:
                    pass
            
            db.create_lesson_session(
                session_id=session_id,
                document_name=filename,
                document_type=filename.split('.')[-1] if '.' in filename else 'text',
                concepts=self.current_state.concepts,
                study_plan=self.current_state.study_plan,
                document_structure=learning_roadmap,
                exam_context=exam_context_data
            )
            
            # Initialize adaptive graph if enabled
            if self.use_adaptive_graph:
                self.adaptive_graph = create_adaptive_graph(self)
                self.adaptive_state = create_initial_state(
                    session_id=session_id,
                    doc_id=self.current_state.doc_id,
                    user_id="default"
                )
                self.adaptive_state["study_plan"] = self.current_state.study_plan
                self.adaptive_state["current_topic"] = self.current_state.study_plan[0]["topic"] if self.current_state.study_plan else ""
                print(f"[WORKFLOW] Adaptive graph initialized for session {session_id}")
            
            # Return initial response
            return {
                "success": True,
                "session_id": session_id,
                "concepts": self.current_state.concepts,
                "study_plan": self.current_state.study_plan,
                "next_action": "start_plan",
                "message": f"Found {len(self.current_state.concepts)} key concepts. Ready to start your {student_choice} session!"
            }
            
        except Exception as e:
            print(f"[WORKFLOW] Error in start_learning_session: {str(e)}")
            import traceback
            print(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": "Session start failed",
                "details": str(e)
            }

    async def start_learning_session_from_saved(self, saved_session_id: str, student_choice: str = "from_beginning") -> Dict[str, Any]:
        """Start a session reusing persisted concepts/segments and existing vector index.
        Does NOT reuse past progress; creates a fresh plan and fresh Redis memory.
        """
        print(f"🚀 Starting session from saved data: {saved_session_id}")
        try:
            # Enforce: clear cross-session summaries when starting from saved (preserve doc index)
            try:
                from app.core.vectorstore import clear_taught_summaries
                clear_taught_summaries()
                print("[WORKFLOW] Cleared taught summaries for start-from-saved")
            except Exception as e:
                print(f"[WORKFLOW] Warning: failed to clear taught summaries on saved: {e}")
            # Load saved lesson session
            saved = db.get_lesson_session(saved_session_id)
            if not saved:
                return {"success": False, "error": "Saved session not found"}

            # Derive doc_id from session_id convention
            doc_id = saved_session_id.replace("session_", "") if saved_session_id.startswith("session_") else saved_session_id
            self.current_state.doc_id = doc_id
            self.current_state.ingest_status = "completed"
            self.current_state.concepts = saved.get("concepts", []) or []
            self.current_state.learning_roadmap = saved.get("document_structure", {}) or {}

            # Initialize FRESH Redis memory for this session (DO NOT reuse old memory!)
            import uuid
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}_{uuid.uuid4().hex[:8]}"
            self.current_state.session_id = session_id
            # Reset any in-memory runtime fields to ensure no carryover
            self.current_state.taught_content = {}
            self.current_state.current_quiz = None
            self.current_state.student_answers = []
            self.current_state.current_step = 0
            self._init_session_memory(
                session_id=session_id,
                metadata={
                    "doc_id": self.current_state.doc_id,
                    "filename": saved.get("document_name", "Unknown"),
                    "loaded_from": saved_session_id,
                    "num_concepts": len(self.current_state.concepts)
                },
                clear_existing=True  # Clear any stale data from previous sessions
            )
            # Populate ONLY segment plans into fresh Redis (no progress/taught restoration)
            if self.memory and self.current_state.concepts:
                try:
                    from typing import List, Dict
                    populated = 0
                    for concept in self.current_state.concepts:
                        topic = concept.get("label")
                        segments: List[Dict] = concept.get("learning_segments") or []
                        if topic and segments:
                            ok = self.memory.store_segment_plan(topic, segments)
                            if ok:
                                populated += 1
                    print(f"[WORKFLOW] Stored segment plans for {populated} topics into fresh session memory")
                except Exception as e:
                    print(f"[WORKFLOW] Failed to store segment plans into memory: {e}")
            
            # Restore exam context from saved session to NEW Redis memory
            if self.memory:
                saved_exam_context = saved.get("exam_context")
                if saved_exam_context:
                    try:
                        self.memory.set_exam_context(
                            exam_type=saved_exam_context.get("exam_type", "general"),
                            exam_details=saved_exam_context.get("exam_details", {})
                        )
                        print(f"[WORKFLOW] Restored exam context: {saved_exam_context.get('exam_type', 'general')}")
                    except Exception as e:
                        print(f"[WORKFLOW] Failed to restore exam context: {e}")

            # Fast path: deterministically build a simple segment-first plan without LLM to reduce latency
            plan: List[Dict[str, Any]] = []
            for concept in (self.current_state.concepts or [])[:6]:  # cap to first 6 for quick start
                topic = concept.get("label", "Topic")
                concept_id = concept.get("concept_id", "concept")
                segments = sorted(concept.get("learning_segments", []), key=lambda s: s.get("order", 0))
                for seg in segments:
                    plan.append({
                        "step_id": f"{concept_id}_{seg.get('segment_id','seg')}",
                        "action": "study_segment",
                        "topic": topic,
                        "concept_id": concept_id,
                        "segment_id": seg.get("segment_id", "seg_1"),
                        "segment_title": seg.get("title", "Segment"),
                        "difficulty": "medium",
                        "est_minutes": seg.get("estimated_minutes", 8),
                        "why_assigned": "Learn this segment"
                    })
                # Do NOT append quiz steps here; planner will trigger quiz only after all segments are completed

            self.current_state.study_plan = plan
            self.current_state.student_choice = student_choice

            # Initialize adaptive graph if enabled
            if self.use_adaptive_graph:
                self.adaptive_graph = create_adaptive_graph(self)
                self.adaptive_state = create_initial_state(
                    session_id=session_id,
                    doc_id=self.current_state.doc_id,
                    user_id="default"
                )
                self.adaptive_state["study_plan"] = self.current_state.study_plan
                self.adaptive_state["current_topic"] = self.current_state.study_plan[0]["topic"] if self.current_state.study_plan else ""
                print(f"[WORKFLOW] Adaptive graph initialized for saved session {session_id}")

            # Return response similar to upload flow (no previous messages)
            return {
                "success": True,
                "session_id": session_id,
                "concepts": self.current_state.concepts,
                "study_plan": self.current_state.study_plan,
                "messages": [],
                "next_action": "start_plan",
                "message": f"Loaded saved concepts ({len(self.current_state.concepts)}) and created a fresh plan."
            }
        except Exception as e:
            print(f"[WORKFLOW] Error in start_learning_session_from_saved: {str(e)}")
            import traceback
            print(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
            return {"success": False, "error": "Failed to start from saved", "details": str(e)}
    
    async def execute_plan_step(self, step_index: int = None) -> Dict[str, Any]:
        """Execute the next step in the study plan"""
        if not self.current_state.study_plan:
            return {
                "success": False,
                "error": "No study plan available"
            }
        
        # Use provided step or current step
        if step_index is None:
            step_index = self.current_state.current_step
        
        # If using adaptive graph, let the graph decide end-of-plan behavior
        if (not self.use_adaptive_graph) and step_index >= len(self.current_state.study_plan):
            return {
                "success": False,
                "error": "All plan steps completed"
            }
        
        # Adaptive graph path (if enabled and initialized)
        if self.use_adaptive_graph and self.adaptive_graph and self.adaptive_state:
            print(f"[WORKFLOW] Using adaptive graph for step execution")
            try:
                # Update adaptive state with current step index
                self.adaptive_state["current_step_index"] = step_index
                
                # If awaiting explicit user action, do not advance automatically
                if self.adaptive_state.get("awaiting_user_next"):
                    print("[WORKFLOW] Awaiting user action; not advancing graph")
                    # Return last result if available, otherwise a minimal prompt
                    last = self.adaptive_state.get("last_result", {})
                    return last or {"success": True, "message": "Ready. Click Next Step to continue."}
                
                # Execute graph (planner decides, then routes to worker)
                updated_state = await self.adaptive_graph.execute_step(self.adaptive_state)
                
                # Sync back to legacy state
                self.current_state.current_step = updated_state["current_step_index"]
                self.adaptive_state = updated_state
                
                # Extract result for API response
                result = updated_state.get("last_result", {})
                if result.get("success"):
                    result["planner_reason"] = updated_state.get("planner_reason", "")
                    # If worker set awaiting_user_next, persist it so we halt next time
                    if updated_state.get("awaiting_user_next"):
                        self.adaptive_state["awaiting_user_next"] = True
                    return result
                else:
                    # If graph returned end/wait state, translate to response
                    next_action = updated_state.get("next_action")
                    if next_action == "wait_for_quiz_submission":
                        # Normalize waiting response to success True
                        normalized = {**(result or {})}
                        normalized["step_type"] = "practice_quiz"
                        normalized["next_action"] = "wait_for_quiz_submission"
                        normalized["success"] = True  # enforce success
                        return normalized
                    elif next_action == "wait_for_answer":
                        normalized = {**(result or {})}
                        normalized["step_type"] = "assessment_wait"
                        normalized["next_action"] = "wait_for_answer"
                        normalized["success"] = True
                        return normalized
                    elif next_action == "complete":
                        return {"success": True, "message": "Learning complete!", "next_action": "complete"}
                    else:
                        return result or {"success": True, "message": "Ready. Click Next Step to continue."}
            except Exception as e:
                print(f"[WORKFLOW] Adaptive graph error, falling back to legacy: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to legacy execution
        
        # Legacy execution path
        step = self.current_state.study_plan[step_index]
        action = step["action"]
        topic = step["topic"]
        concept_id = step.get("concept_id")
        
        # Memory-driven simplification: prefer segment-based study when segment plan exists
        if action in ("study_topic", "study_segment") and self.memory:
            try:
                next_seg = self.memory.get_next_segment(topic)
                if next_seg:
                    # Ensure step uses study_segment with concrete segment_id/title
                    step["action"] = "study_segment"
                    step["segment_id"] = next_seg.get("segment_id", "seg_1")
                    step["segment_title"] = next_seg.get("title", "Segment")
                    action = step["action"]
            except Exception:
                pass

        print(f"🎯 Executing step {step_index + 1}: {action} for {topic}")
        
        try:
            result = None
            if action == "diagnostic_quiz":
                result = await self._execute_diagnostic_quiz(step, concept_id)
            elif action == "calibration_quiz":
                result = await self._execute_calibration_quiz(step, concept_id)
            elif action == "study_topic":
                result = await self._execute_study_topic(step, concept_id)
            elif action == "study_segment":
                result = await self._execute_study_segment(step, concept_id)
            elif action == "practice_quiz":
                # Enforce: do not quiz until all segments for topic are completed
                print(f"[WORKFLOW] Practice quiz encountered at step {self.current_state.current_step} for topic={topic}")
                if self.memory:
                    try:
                        remaining_next = self.memory.get_next_segment(topic)
                        print(f"[WORKFLOW] Memory next segment for topic '{topic}': {remaining_next}")
                        if remaining_next:
                            # Insert the next segment step before quiz and execute it now
                            print(f"[WORKFLOW] Injecting remaining segment before quiz: {remaining_next}")
                            insert = {
                                "step_id": f"auto_{remaining_next.get('segment_id', 'seg')}",
                                "action": "study_segment",
                                "topic": topic,
                                "concept_id": concept_id,
                                "segment_id": remaining_next.get("segment_id", "seg_1"),
                                "segment_title": remaining_next.get("title", "Segment"),
                                "difficulty": "medium",
                                "est_minutes": remaining_next.get("estimated_minutes", 8),
                                "why_assigned": "Complete remaining segment before topic-level quiz"
                            }
                            try:
                                upcoming = self.current_state.study_plan[self.current_state.current_step:self.current_state.current_step+2]
                                print(f"[WORKFLOW] Upcoming before insert: {upcoming}")
                            except Exception:
                                pass
                            self.current_state.study_plan.insert(self.current_state.current_step, insert)
                            try:
                                upcoming2 = self.current_state.study_plan[self.current_state.current_step:self.current_state.current_step+2]
                                print(f"[WORKFLOW] Upcoming after insert: {upcoming2}")
                            except Exception:
                                pass
                            step = insert
                            action = "study_segment"
                            result = await self._execute_study_segment(step, concept_id)
                        else:
                            print(f"[WORKFLOW] No remaining segments. Proceeding to topic-level quiz for {topic}")
                            result = await self._execute_practice_quiz(step, concept_id)
                    except Exception as e:
                        print(f"[WORKFLOW] Error checking remaining segments: {e}. Proceeding to quiz")
                        result = await self._execute_practice_quiz(step, concept_id)
                else:
                    print(f"[WORKFLOW] No memory available. Proceeding to quiz for {topic}")
                    result = await self._execute_practice_quiz(step, concept_id)
            elif action == "review_results":
                result = await self._execute_review_results(step)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
            
            # Log what we're returning
            try:
                print(f"[WORKFLOW] Returning result with keys: {list(result.keys())}")
                if 'content' in result and isinstance(result.get('content'), str):
                    print(f"[WORKFLOW] Content preview: {result['content'][:100]}...")
                else:
                    print(f"[WORKFLOW] WARNING: No 'content' field in result! Full result: {result}")
            except Exception:
                pass
            # Also write step result to session log
            try:
                logger = get_logger()
                logger.log_agent_action("WorkflowOrchestrator", "execute_plan_step", f"action={action}, topic={topic}")
                logger.log_data("step_result", result)
            except Exception:
                pass
            
            return result
                
        except Exception as e:
            print(f"[WORKFLOW] Exception: {str(e)}")
            import traceback
            print(f"[WORKFLOW] Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Step execution failed: {str(e)}"
            }
    
    async def _execute_diagnostic_quiz(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute diagnostic quiz step"""
        topic = step["topic"]
        
        # Generate quiz
        quiz_result = await self.tutor_evaluator.execute(
            "generate_quiz",
            topic=topic,
            concept_id=concept_id,
            difficulty="medium",
            num_questions=2
        )
        
        if not quiz_result.success:
            return {
                "success": False,
                "error": "Quiz generation failed",
                "details": quiz_result.reasoning
            }
        
        quiz = quiz_result.data
        self.current_state.current_quiz = quiz
        
        return {
            "success": True,
            "step_type": "diagnostic_quiz",
            "quiz": quiz,
            "instructions": f"Answer these {len(quiz['questions'])} questions about {topic}. This is a quick diagnostic to assess your current level.",
            "next_action": "submit_answers"
        }
    
    async def _execute_calibration_quiz(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute calibration quiz step"""
        topic = step["topic"]
        
        # Generate easy quiz for calibration
        quiz_result = await self.tutor_evaluator.execute(
            "generate_quiz",
            topic=topic,
            concept_id=concept_id,
            difficulty="easy",
            num_questions=1
        )
        
        if not quiz_result.success:
            return {
                "success": False,
                "error": "Quiz generation failed",
                "details": quiz_result.reasoning
            }
        
        quiz = quiz_result.data
        self.current_state.current_quiz = quiz
        
        return {
            "success": True,
            "step_type": "calibration_quiz",
            "quiz": quiz,
            "instructions": f"Answer this quick question about {topic} to help me calibrate the difficulty level for you.",
            "next_action": "submit_answers"
        }
    
    async def _execute_study_topic(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute study topic step with automatic introduction"""
        topic = step["topic"]
        print(f"[WORKFLOW] Executing study_topic for: {topic}")
        
        # Update current topic in state
        self.current_state.current_topic = topic
        
        # IMPROVED RETRIEVAL: Use LLM-generated queries if available, otherwise fallback
        print(f"[WORKFLOW] Retrieving content for topic: {topic}")
        
        # Try to get LLM-generated queries from concept
        concept_queries = None
        for concept in self.current_state.concepts:
            if concept.get("label") == topic or concept.get("concept_id") == concept_id:
                concept_queries = concept.get("search_queries", [])
                break
        
        # Use LLM queries if available, otherwise generate heuristic queries
        if concept_queries and len(concept_queries) > 0:
            queries = concept_queries[:4]  # Use up to 4 LLM-generated queries
            print(f"[WORKFLOW] Using {len(queries)} LLM-generated search queries")
        else:
            queries = self._generate_search_queries(topic)
            print(f"[WORKFLOW] Using {len(queries)} heuristic search queries")
        
        # Retrieve with all queries and deduplicate
        all_documents = []
        all_metadatas = []
        seen_texts = set()
        
        for query in queries:
            # Use hybrid retrieval (vector + BM25). Falls back to vector if BM25 unavailable.
            results = vectorstore.query_hybrid(query, k=5, alpha=0.6)
            docs = results.get('documents', [[]])[0] if 'documents' in results else []
            metas = results.get('metadatas', [[]])[0] if 'metadatas' in results else []
            
            for doc, meta in zip(docs, metas):
                if doc not in seen_texts:
                    seen_texts.add(doc)
                    all_documents.append(doc)
                    all_metadatas.append(meta)
        
        documents = all_documents
        print(f"[WORKFLOW] Retrieved {len(documents)} unique documents for {topic}")
        # Log retrieved chunks for evaluation
        try:
            logger = get_logger()
            preview = [
                {
                    "text_preview": (doc[:220] + "...") if isinstance(doc, str) and len(doc) > 220 else doc,
                    "metadata": meta
                }
                for doc, meta in zip(all_documents[:10], all_metadatas[:10])
            ]
            logger.log_data("retrieved_chunks", {
                "topic": topic,
                "queries": queries,
                "count": len(documents),
                "samples": preview
            })
        except Exception:
            pass
        
        if not documents or len(documents) == 0:
            # Fallback to regular search
            print(f"[WORKFLOW] No documents with filter, trying regular search")
            results = vectorstore.query_top_k(topic, k=5)
        documents = results.get('documents', [[]])[0] if 'documents' in results else []
        
        if not documents or len(documents) == 0:
            error_msg = f"I couldn't find detailed content about {topic} in the uploaded document. Let me explain based on general knowledge."
            return {
                "success": True,
                "step_type": "study_topic",
                "topic": topic,
                "study_content": "",
                "content": error_msg,
                "instructions": "Ask me specific questions about this topic!",
                "next_action": "ask_questions_or_continue"
            }
        
        # Format study material with proper context (use more chunks for better context)
        # Sanitize: remove heading-only lines and keep substantive sentences
        def clean_snippet(s: str) -> str:
            import re
            lines = [ln.strip() for ln in s.splitlines()]
            cleaned = []
            for ln in lines:
                # Skip very short heading-like lines
                if len(ln) < 25 and (ln.lower().startswith('section:') or ln.istitle()):
                    continue
                cleaned.append(ln)
            text = ' '.join(cleaned)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        cleaned_docs = [clean_snippet(d) for d in documents[:6] if isinstance(d, str) and d.strip()]
        study_content = "\n\n---\n\n".join([d for d in cleaned_docs if len(d) > 120])
        
        # Generate grounded, structured explanation using LLM (with timeout)
        print(f"[WORKFLOW] Generating grounded explanation for {topic}")
        recent_summaries = []
        if self.memory:
            try:
                recent_summaries = self.memory.get_last_session_summaries(k=3)
            except Exception:
                recent_summaries = []
        educator_prompt = f"""Explain this topic using the provided context.

            TOPIC: {topic}

CONTEXT:
{study_content[:1500]}

RECENT SESSION (for continuity):
{json.dumps(recent_summaries)[:600]}

Return ONLY JSON (no code fences):
{{
  "full_text": "4-6 paragraph explanation grounded in context",
  "summary": "1-2 sentence summary",
  "excerpts": ["quote from context"],
  "student_facing_next_steps": ["exercise 1"],
  "follow_up_questions": ["question 1"],
  "memory_patch": {{"session_summary_delta": "note", "confidence": 0.8}}
}}"""
        
        
        try:
            import asyncio
            # Add 10 second timeout
            start_time = time.time()
            intro_response = await asyncio.wait_for(
                self.llm.ainvoke(educator_prompt),
                timeout=60.0  # Increased to 60s; simplified prompt should reduce actual time
            )
            duration = time.time() - start_time
            # Try to parse JSON contract; fallback to plain text
            parsed = None
            try:
                content_raw = intro_response.content.strip()
                if content_raw.startswith("```"):
                    if '```json' in content_raw:
                        content_raw = content_raw.split('```json', 1)[1]
                    parts = content_raw.split('```')
                    if len(parts) >= 2:
                        content_raw = parts[1]
                if not content_raw.strip().startswith('{'):
                    import re
                    m = re.search(r"\{[\s\S]*\}", content_raw)
                    if m:
                        content_raw = m.group(0)
                parsed = json.loads(content_raw)
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("full_text"):
                grounded_explanation = parsed.get("full_text", "")
                summary_text = parsed.get("summary", "")
                memory_patch = parsed.get("memory_patch", {})
                # Store structured taught segment
                if self.memory:
                    try:
                        self.memory.store_taught_segment_json(
                            topic,
                            "full_topic",
                            {
                                "full_text": grounded_explanation,
                                "summary": summary_text,
                                "excerpts": parsed.get("excerpts", []),
                                "sources": parsed.get("sources", []),
                                "segment_id": "full_topic",
                                "topic": topic
                            }
                        )
                    except Exception:
                        pass
            else:
                grounded_explanation = intro_response.content
                # Fallback summary: first 180 chars
                summary_text = (intro_response.content or "").strip()[:180]
                memory_patch = {}
            
            # Log LLM call
            print(f"\n{'='*80}")
            print(f"🤖 LLM CALL - Generating Grounded Explanation")
            print(f"{'='*80}")
            print(f"Topic: {topic}")
            print(f"Duration: {duration:.2f}s")
            print(f"Prompt (first 200 chars): {educator_prompt[:200]}...")
            print(f"Response (first 200 chars): {grounded_explanation[:200]}...")
            print(f"{'='*80}\n")
            
            print(f"[WORKFLOW] Grounded explanation generated successfully")
        except asyncio.TimeoutError:
            print(f"[WORKFLOW] Explanation generation timed out, using fallback")
            grounded_explanation = (
                f"**{topic}**\n\n"
                f"Here is a concise overview based on the document.\n\n"
                f"- What it is: [INFERRED] A high-level summary will appear here.\n"
                f"- Why it matters: [INFERRED] Brief motivation.\n\n"
                f"Key takeaways:\n- [INFERRED] Takeaway 1\n- [INFERRED] Takeaway 2\n- [INFERRED] Takeaway 3\n- [INFERRED] Takeaway 4\n- [INFERRED] Takeaway 5\n\n"
                f"Examples:\n1) [INFERRED] Short example.\n2) [INFERRED] Short example.\n\n"
                f"Practice questions:\n1) Write a short question.\n2) Write a short question.\n\n"
                f"Study plan (next steps):\n1) Skim the section.\n2) Work an example.\n3) Answer practice questions.\n"
            )
        except Exception as e:
            print(f"[WORKFLOW] Error generating explanation: {e}")
            grounded_explanation = f"**{topic}**\n\n[INFERRED] Let's explore this concept step by step using the document context."
        
        # Use normalized explanation (extract full_text from JSON if needed)
        full_explanation = self._extract_full_text_from_possible_json(grounded_explanation)
        
        # STEP 1 FIX: Store taught content for quiz grounding
        self.current_state.taught_content[topic] = full_explanation
        print(f"[WORKFLOW] Stored taught content for {topic} ({len(full_explanation)} chars)")
        
        # Store in Redis memory if available (for segment-based teaching)
        if self.memory:
            # For now, store as a single segment since we're still using topic-based teaching
            # Later this will be updated to store per-segment content
            self.memory.store_taught_segment(topic, "full_topic", full_explanation)
            self.memory.update_context({
                "current_topic": topic,
                "current_segment": "full_topic",
                "last_taught_content": f"{topic}:full_topic"
            })
            # Ensure last_k summary is pushed even if model omitted it
            try:
                if summary_text:
                    self.memory.push_session_summary({
                        "segment_id": "full_topic",
                        "topic": topic,
                        "summary": summary_text,
                        "timestamp": time.time(),
                        "memory_delta": memory_patch.get("session_summary_delta") if isinstance(memory_patch, dict) else None
                    }, k=3)
                # Write to long-term summary store (Chroma) for semantic recall
                vectorstore.upsert_taught_summary(
                    _id=f"{self.current_state.doc_id}:{topic}:full_topic",
                    text=summary_text,
                    metadata={"doc_id": self.current_state.doc_id, "topic": topic, "segment_id": "full_topic"}
                )
            except Exception:
                pass
            # Push last-k summary if available
            try:
                if summary_text:
                    self.memory.push_session_summary({
                        "segment_id": "full_topic",
                        "topic": topic,
                        "summary": summary_text,
                        "timestamp": time.time(),
                        "memory_delta": memory_patch.get("session_summary_delta")
                    }, k=3)
            except Exception:
                pass
        
        # Log the study content in lesson history
        if self.current_state.doc_id:
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=full_explanation,
                sources=[f"Document: {self.current_state.doc_id}"]
            )
        
        print(f"[WORKFLOW] Successfully prepared study content for {topic}")
        
        return {
            "success": True,
            "step_type": "study_topic",
            "topic": topic,
            "study_content": study_content,
            "content": full_explanation,  # This will be displayed in chat automatically
            "exercises": parsed.get("student_facing_next_steps", []) if isinstance(parsed, dict) else [],
            "follow_ups": parsed.get("follow_up_questions", []) if isinstance(parsed, dict) else [],
            "instructions": "Feel free to ask me any questions about this topic!",
            "next_action": "ask_questions_or_continue"
        }
    
    async def _execute_study_segment(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute study segment step - focused teaching of one segment"""
        topic = step["topic"]
        segment_id = step.get("segment_id")
        segment_title = step.get("segment_title") or step.get("title")
        # If missing, fill from memory's next segment
        if (not segment_id or not segment_title) and self.memory:
            try:
                next_seg = self.memory.get_next_segment(topic)
                if next_seg:
                    segment_id = segment_id or next_seg.get("segment_id", "seg_1")
                    segment_title = segment_title or next_seg.get("title", "Segment")
                    step["segment_id"] = segment_id
                    step["segment_title"] = segment_title
                    print(f"[WORKFLOW] Filled missing segment from memory: id={segment_id} title={segment_title}")
            except Exception as e:
                print(f"[WORKFLOW] Could not fill missing segment: {e}")
        # Final fallback
        segment_id = segment_id or "seg_1"
        segment_title = segment_title or "Segment"
        
        print(f"[WORKFLOW] Executing study_segment for: {topic} - {segment_title}")
        
        # Update current topic in state
        self.current_state.current_topic = topic
        
        # Get segment details from concept
        segment_info = None
        for concept in self.current_state.concepts:
            if concept.get("concept_id") == concept_id:
                segments = concept.get("learning_segments", [])
                segment_info = next((s for s in segments if s.get("segment_id") == segment_id), None)
                break
        
        if not segment_info:
            print(f"[WORKFLOW] Segment {segment_id} not found, falling back to topic teaching")
            return await self._execute_study_topic(step, concept_id)
        
        # Get segment-specific search queries
        concept_queries = None
        for concept in self.current_state.concepts:
            if concept.get("concept_id") == concept_id:
                concept_queries = concept.get("search_queries", [])
                break
        
        # Use LLM queries if available, otherwise generate heuristic queries
        if concept_queries and len(concept_queries) > 0:
            queries = concept_queries[:4]  # Use up to 4 LLM-generated queries
            print(f"[WORKFLOW] Using {len(queries)} LLM-generated search queries for segment")
        else:
            # Generate segment-specific queries
            queries = [
                f"{topic} {segment_title}",
                f"{segment_title} explanation examples",
                f"{topic} {segment_title} key concepts"
            ]
            # Add objectives-based query terms to strengthen retrieval
            objectives = segment_info.get('learning_objectives', [])
            if isinstance(objectives, list) and objectives:
                primary_obj = objectives[0]
                if isinstance(primary_obj, str) and primary_obj.strip():
                    queries.append(f"{topic} {segment_title} {primary_obj}")
                if len(objectives) > 1 and isinstance(objectives[1], str):
                    queries.append(f"{segment_title} {objectives[1]}")
            print(f"[WORKFLOW] Using {len(queries)} heuristic search queries for segment")
        
        # Retrieve with all queries and deduplicate
        all_documents = []
        all_metadatas = []
        seen_texts = set()
        
        for query in queries:
            # Use hybrid retrieval (vector + BM25). Falls back to vector if BM25 unavailable.
            results = vectorstore.query_hybrid(query, k=5, alpha=0.6)
            docs = results.get('documents', [[]])[0] if 'documents' in results else []
            metas = results.get('metadatas', [[]])[0] if 'metadatas' in results else []
            
            for doc, meta in zip(docs, metas):
                if doc not in seen_texts:
                    seen_texts.add(doc)
                    all_documents.append(doc)
                    all_metadatas.append(meta)
        
        documents = all_documents
        print(f"[WORKFLOW] Retrieved {len(documents)} unique documents for segment {segment_title}")
        
        # Log retrieved chunks for evaluation
        try:
            logger = get_logger()
            preview = [
                {
                    "text_preview": (doc[:220] + "...") if isinstance(doc, str) and len(doc) > 220 else doc,
                    "metadata": meta
                }
                for doc, meta in zip(all_documents[:10], all_metadatas[:10])
            ]
            logger.log_data("retrieved_chunks", {
                "topic": topic,
                "segment": segment_title,
                "queries": queries,
                "count": len(documents),
                "samples": preview
            })
        except Exception:
            pass
        
        if not documents or len(documents) == 0:
            error_msg = f"I couldn't find detailed content about {segment_title} in the uploaded document. Let me explain based on general knowledge."
            return {
                "success": True,
                "step_type": "study_segment",
                "topic": topic,
                "segment_id": segment_id,
                "segment_title": segment_title,
                "study_content": "",
                "content": error_msg,
                "instructions": "Ask me specific questions about this segment!",
                "next_action": "ask_questions_or_continue"
            }
        
        # Format study material with proper context (use more chunks for better context)
        def clean_snippet(s: str) -> str:
            import re
            lines = [ln.strip() for ln in s.splitlines()]
            cleaned = []
            for ln in lines:
                # Skip very short heading-like lines
                if len(ln) < 25 and (ln.lower().startswith('section:') or ln.istitle()):
                    continue
                cleaned.append(ln)
            text = ' '.join(cleaned)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        cleaned_docs = [clean_snippet(d) for d in documents[:6] if isinstance(d, str) and d.strip()]
        # Drop snippets that look like headings-only lines
        cleaned_docs = [d for d in cleaned_docs if not d.lower().startswith('section:') or len(d.split()) > 25]
        study_content = "\n\n---\n\n".join([d for d in cleaned_docs if len(d) > 120])
        
        # Generate segment-specific explanation using LLM
        print(f"[WORKFLOW] Generating explanation for segment: {segment_title}")
        recent_summaries = []
        if self.memory:
            try:
                recent_summaries = self.memory.get_last_session_summaries(k=3)
            except Exception:
                recent_summaries = []
        
        # Get exam context for tailored teaching
        exam_context = {}
        exam_style_instruction = ""
        if self.memory:
            try:
                exam_context = self.memory.get_exam_context()
                exam_type = exam_context.get("exam_type", "general")
                
                # Create exam-specific teaching instructions
                if exam_type == "JEE":
                    exam_style_instruction = "\n**EXAM FOCUS: JEE (Technical Depth)**\n- Emphasize mathematical rigor and derivations\n- Include numerical problem-solving approaches\n- Focus on concept application in physics/chemistry/math\n- Use technical terminology precisely\n"
                elif exam_type == "UPSC":
                    exam_style_instruction = "\n**EXAM FOCUS: UPSC (Broad Coverage)**\n- Provide comprehensive conceptual understanding\n- Include real-world applications and current affairs connections\n- Cover breadth of topic with multiple perspectives\n- Use clear, articulate language suitable for essays\n"
                elif exam_type in ["SAT", "GRE"]:
                    exam_style_instruction = f"\n**EXAM FOCUS: {exam_type}**\n- Balance conceptual clarity with problem-solving\n- Include standardized test-style examples\n- Focus on reasoning and analytical skills\n"
                elif exam_type != "general":
                    exam_style_instruction = f"\n**EXAM FOCUS: {exam_type}**\n- Tailor explanations for {exam_type} preparation\n- Include exam-relevant examples and applications\n"
            except Exception as e:
                print(f"[WORKFLOW] Could not get exam context: {e}")
        
        # Get current difficulty level for adaptive teaching
        avg_difficulty = 3.0  # Neutral default
        if self.memory:
            try:
                avg_difficulty = self.memory.get_avg_difficulty_for_topic(topic)
            except Exception:
                pass
        
        # Adjust teaching complexity based on difficulty feedback
        complexity_instruction = ""
        if avg_difficulty <= 2.0:
            complexity_instruction = "\n**ADJUST DIFFICULTY: Student finds material challenging**\n- Use simpler language and more examples\n- Break down complex ideas into smaller steps\n- Add more analogies and visual descriptions\n"
        elif avg_difficulty >= 4.5:
            complexity_instruction = "\n**ADJUST DIFFICULTY: Student finds material easy**\n- Increase technical depth and rigor\n- Add advanced concepts and edge cases\n- Challenge with thought-provoking questions\n"
        
        # Build progressive narrative context
        narrative_context = ""
        if recent_summaries:
            narrative_context = "WHAT WE'VE COVERED SO FAR:\n"
            for i, summ in enumerate(recent_summaries[:3]):
                summary_text = summ.get('summary', '') if isinstance(summ, dict) else str(summ)
                narrative_context += f"{i+1}. {summary_text}\n"
            narrative_context += "\nNOW LET'S BUILD ON THIS:\n"
        
        segment_prompt = f"""Teach this segment using the provided context.{exam_style_instruction}{complexity_instruction}

        {narrative_context}
        TOPIC: {topic}
SEGMENT: {segment_title}
OBJECTIVES: {segment_info.get('learning_objectives', [])}

CONTEXT:
{study_content[:1500]}

Return ONLY JSON (no code fences):
{{
  "full_text": "3-4 paragraph explanation. Connect to previous material if relevant.",
  "summary": "1-2 sentence summary",
  "excerpts": ["quote from context"],
  "student_facing_next_steps": ["exercise 1", "exercise 2"],
  "follow_up_questions": ["question 1"],
  "memory_patch": {{"session_summary_delta": "brief note", "confidence": 0.8}}
}}"""
        
        # Initialize parsed to None at the start
        parsed = None
        summary_text = ""
        memory_patch = {}
        
        try:
            import asyncio
            start_time = time.time()
            segment_response = await asyncio.wait_for(
                self.llm.ainvoke(segment_prompt),
                timeout=60.0  # Increased to 60s for safety; simplified prompt should reduce actual time
            )
            duration = time.time() - start_time
            
            # Parse JSON response
            parsed = None
            try:
                content_raw = segment_response.content.strip()
                # Remove code fences if present
                if content_raw.startswith("```"):
                    if '```json' in content_raw:
                        content_raw = content_raw.split('```json', 1)[1]
                    parts = content_raw.split('```')
                    if len(parts) >= 2:
                        content_raw = parts[1].strip()
                
                # Extract JSON object if wrapped in text
                if not content_raw.startswith('{'):
                    import re
                    m = re.search(r"\{[\s\S]*\}", content_raw)
                    if m:
                        content_raw = m.group(0)
                
                parsed = json.loads(content_raw)
            except json.JSONDecodeError as e:
                print(f"[WORKFLOW] JSON parse error: {e}")
                print(f"[WORKFLOW] Raw response (first 500 chars): {content_raw[:500]}")
                parsed = None
            except Exception as e:
                print(f"[WORKFLOW] Parse error: {e}")
                parsed = None
            
            # Extract content from parsed JSON or use raw response
            if isinstance(parsed, dict) and parsed.get("full_text"):
                segment_explanation = parsed.get("full_text", "")
                summary_text = parsed.get("summary", "")
                memory_patch = parsed.get("memory_patch", {})
                
                # Store structured taught segment
                if self.memory:
                    try:
                        self.memory.store_taught_segment_json(
                            topic,
                            segment_id,
                            {
                                "full_text": segment_explanation,
                                "summary": summary_text,
                                "excerpts": parsed.get("excerpts", []),
                                "sources": parsed.get("sources", []),
                                "segment_id": segment_id,
                                "segment_title": segment_title,
                                "topic": topic
                            }
                        )
                    except Exception as e:
                        print(f"[WORKFLOW] Failed to store segment JSON: {e}")
            else:
                # Fallback: use raw content
                print(f"[WORKFLOW] JSON parsing failed, using raw content")
                segment_explanation = segment_response.content
                summary_text = (segment_response.content or "").strip()[:180]
                memory_patch = {}
                parsed = None  # Ensure parsed is None for later checks
            
            # Log LLM call
            print(f"\n{'='*80}")
            print(f"🤖 LLM CALL - Generating Segment Explanation")
            print(f"{'='*80}")
            print(f"Topic: {topic}")
            print(f"Segment: {segment_title}")
            print(f"Duration: {duration:.2f}s")
            print(f"JSON Parsed: {'Yes' if parsed else 'No (using raw content)'}")
            print(f"Content Length: {len(segment_explanation)} chars")
            print(f"Response (first 200 chars): {segment_explanation[:200]}...")
            print(f"{'='*80}\n")
            
            print(f"[WORKFLOW] Segment explanation generated successfully")
            
        except asyncio.TimeoutError:
            print(f"[WORKFLOW] ⚠️ Segment explanation generation timed out (>45s), using fallback")
            segment_explanation = (
                f"**{segment_title}**\n\n"
                f"This segment focuses on key concepts in {topic}.\n\n"
                f"**Learning Objectives:**\n" +
                "\n".join([f"- {obj}" for obj in segment_info.get('learning_objectives', ['Understand this concept'])]) +
                f"\n\n**Key Points:**\n"
                f"- This topic is essential for understanding {topic}\n"
                f"- Focus on the core concepts and their applications\n"
                f"- Practice with examples to reinforce learning\n"
            )
            parsed = None
            summary_text = f"{segment_title} - Core concepts"
            
        except Exception as e:
            print(f"[WORKFLOW] ❌ Error generating segment explanation: {e}")
            import traceback
            traceback.print_exc()
            segment_explanation = (
                f"**{segment_title}**\n\n"
                f"Let's explore this topic step by step.\n\n"
                f"**Learning Objectives:**\n" +
                "\n".join([f"- {obj}" for obj in segment_info.get('learning_objectives', ['Understand this concept'])])
            )
            parsed = None
            summary_text = segment_title
        
        # Use normalized segment explanation
        full_explanation = self._extract_full_text_from_possible_json(segment_explanation)
        
        # STEP 1 FIX: Store taught content for quiz grounding (with segment info)
        segment_key = f"{topic}:{segment_id}"
        self.current_state.taught_content[segment_key] = full_explanation
        print(f"[WORKFLOW] Stored taught content for {segment_key} ({len(full_explanation)} chars)")
        
        # Store in Redis memory if available (for segment-based teaching)
        if self.memory:
            self.memory.store_taught_segment(topic, segment_id, full_explanation)
            self.memory.mark_segment_started(topic, segment_id)
            self.memory.update_context({
                "current_topic": topic,
                "current_segment": segment_id,
                "last_taught_content": f"{topic}:{segment_id}"
            })
            # Ensure last_k summary is pushed even if model omitted it
            try:
                if summary_text:
                    self.memory.push_session_summary({
                        "segment_id": segment_id,
                        "topic": topic,
                        "summary": summary_text,
                        "timestamp": time.time(),
                        "memory_delta": memory_patch.get("session_summary_delta") if isinstance(memory_patch, dict) else None
                    }, k=3)
                vectorstore.upsert_taught_summary(
                    _id=f"{self.current_state.doc_id}:{topic}:{segment_id}",
                    text=summary_text,
                    metadata={"doc_id": self.current_state.doc_id, "topic": topic, "segment_id": segment_id}
                )
            except Exception:
                pass
            # Push last-k summary if available
            try:
                if summary_text:
                    self.memory.push_session_summary({
                        "segment_id": segment_id,
                        "topic": topic,
                        "summary": summary_text,
                        "timestamp": time.time(),
                        "memory_delta": memory_patch.get("session_summary_delta")
                    }, k=3)
            except Exception:
                pass
        
            # Mark this segment completed to allow planner to advance; quiz only after all segments
            try:
                ok = self.memory.mark_segment_completed(topic, segment_id)
                if ok:
                    print(f"[WORKFLOW] Marked segment completed: {topic}/{segment_id}")
            except Exception as e:
                print(f"[WORKFLOW] Failed to mark segment completed: {e}")

        # Log the study content in lesson history
        if self.current_state.doc_id:
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=full_explanation,
                sources=[f"Document: {self.current_state.doc_id}"]
            )
        
        print(f"[WORKFLOW] Successfully prepared segment content for {segment_title}")
        
        # If this was the last segment for the topic, insert a topic-level quiz next
        try:
            if self.memory:
                remaining_next = self.memory.get_next_segment(topic)
                if not remaining_next:
                    # Check if a practice_quiz for this topic is already scheduled next
                    insert_index = self.current_state.current_step + 1
                    already_has_quiz = False
                    if insert_index < len(self.current_state.study_plan):
                        nxt = self.current_state.study_plan[insert_index]
                        already_has_quiz = nxt.get("action") == "practice_quiz" and nxt.get("topic") == topic
                    if not already_has_quiz:
                        self.current_state.study_plan.insert(insert_index, {
                            "step_id": f"quiz_{topic}",
                            "action": "practice_quiz",
                            "topic": topic,
                            "concept_id": concept_id,
                            "difficulty": "medium",
                            "est_minutes": 6,
                            "why_assigned": f"Topic-level quiz for {topic} after all segments"
                        })
        except Exception:
            pass

        # Don't add text-based difficulty request - frontend will show buttons instead

        return {
            "success": True,
            "step_type": "study_segment",
            "topic": topic,
            "segment_id": segment_id,
            "segment_title": segment_title,
            "study_content": study_content,
            "content": full_explanation,  # No difficulty request text - use buttons instead
            "exercises": parsed.get("student_facing_next_steps", []) if isinstance(parsed, dict) else [],
            "follow_ups": parsed.get("follow_up_questions", []) if isinstance(parsed, dict) else [],
            "instructions": f"Feel free to ask me any questions about {segment_title}!",
            "next_action": "ask_questions_or_continue"
        }
    
    async def _execute_practice_quiz(self, step: Dict, concept_id: str) -> Dict[str, Any]:
        """Execute practice quiz step with notification"""
        topic = step["topic"]
        print(f"[WORKFLOW] Executing practice_quiz for: {topic}")
        
        # Update current topic
        self.current_state.current_topic = topic
        
        # Topic-level quiz: aggregate taught content across all segments under the topic
        segment_id = step.get("segment_id")  # optional
        taught_content = ""

        if self.memory:
            # Prefer structured JSONs, concatenate full_texts in order (most recent first reasonable for MVP)
            taught_jsons = self.memory.get_taught_segments_for_topic(topic)
            if taught_jsons:
                texts = [j.get("full_text", "") for j in taught_jsons if j.get("full_text")]
                taught_content = "\n\n---\n\n".join(texts)
        # Fallbacks
        if not taught_content:
            if segment_id:
                segment_key = f"{topic}:{segment_id}"
                taught_content = self.current_state.taught_content.get(segment_key, "") or (
                    self.memory.get_taught_segment(topic, segment_id) if self.memory else ""
                )
            else:
                taught_content = self.current_state.taught_content.get(topic, "") or (
                    self.memory.get_taught_segment(topic, "full_topic") if self.memory else ""
                )
        print(f"[WORKFLOW] Using {len(taught_content)} chars of taught content for topic quiz grounding")
        
        # Determine quiz difficulty based on student's performance and ratings
        quiz_difficulty = "medium"  # Default
        if self.memory:
            try:
                avg_difficulty = self.memory.get_avg_difficulty_for_topic(topic)
                recent_quizzes = self.memory.get_recent_quiz_results(topic, k=2)
                
                # Calculate average quiz performance
                avg_quiz_score = 0.0
                if recent_quizzes:
                    scores = [q.get("score", 0) for q in recent_quizzes]
                    avg_quiz_score = sum(scores) / len(scores) if scores else 0.0
                
                # Adaptive difficulty logic
                if avg_difficulty >= 4.5 or avg_quiz_score >= 0.85:
                    quiz_difficulty = "hard"  # Student finds it easy or performs well
                    print(f"[WORKFLOW] Increasing quiz difficulty to HARD (avg_rating={avg_difficulty:.1f}, avg_score={avg_quiz_score:.2f})")
                elif avg_difficulty <= 2.0 or avg_quiz_score <= 0.5:
                    quiz_difficulty = "easy"  # Student struggles
                    print(f"[WORKFLOW] Decreasing quiz difficulty to EASY (avg_rating={avg_difficulty:.1f}, avg_score={avg_quiz_score:.2f})")
                else:
                    quiz_difficulty = "medium"
                    print(f"[WORKFLOW] Keeping quiz difficulty MEDIUM (avg_rating={avg_difficulty:.1f}, avg_score={avg_quiz_score:.2f})")
            except Exception as e:
                print(f"[WORKFLOW] Could not determine adaptive difficulty: {e}")
        
        # Get exam context for quiz style
        exam_context = {}
        if self.memory:
            try:
                exam_context = self.memory.get_exam_context()
            except Exception:
                pass
        
        # Generate practice quiz
        print(f"[WORKFLOW] Preparing to call tutor_evaluator.generate_quiz | difficulty={quiz_difficulty}")
        quiz_result = await self.tutor_evaluator.execute(
            "generate_quiz",
            topic=topic,
            concept_id=concept_id,
            difficulty=quiz_difficulty,  # Dynamic difficulty
            num_questions=3,
            taught_content=taught_content,  # NEW: Pass what was taught
            segment_id=segment_id,  # NEW: Pass segment info
            exam_context=exam_context  # NEW: Pass exam context for question style
        )
        
        if not quiz_result.success:
            return {
                "success": False,
                "error": "Quiz generation failed",
                "details": quiz_result.reasoning
            }
        
        quiz = quiz_result.data
        # Diagnostics: validate quiz structure
        try:
            if not isinstance(quiz, dict):
                raise ValueError(f"Quiz is not a dict: {type(quiz)}")
            if "questions" not in quiz or not isinstance(quiz["questions"], list):
                raise ValueError("Quiz missing 'questions' array")
            print(f"[WORKFLOW] Quiz OK | questions={len(quiz['questions'])} | has_intro={bool(quiz.get('intro'))}")
            if quiz.get('questions'):
                q0 = quiz['questions'][0]
                print(f"[WORKFLOW] Q1 type={q0.get('type')} | options={len(q0.get('options', [])) if isinstance(q0.get('options'), list) else 'na'}")
        except Exception as e:
            print(f"[WORKFLOW] Quiz validation error: {e}")
        self.current_state.current_quiz = quiz
        
        # Generate quiz notification message
        quiz_notification = f"📝 **Quiz Time: {topic}**\n\nI've prepared {len(quiz['questions'])} practice questions to help you master this topic. Take your time and think through each one carefully.\n\nClick the quiz button below to begin!"
        
        # Log quiz notification in lesson history
        if self.current_state.doc_id:
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=quiz_notification,
                sources=[]
            )
        
        # Final safeguard: ensure quiz is JSON-serializable
        try:
            import json
            json.dumps(quiz)
        except Exception as e:
            print(f"[WORKFLOW] Quiz not JSON-serializable: {e}; attempting to sanitize")
            try:
                def to_jsonable(x):
                    if isinstance(x, (str, int, float, bool)) or x is None:
                        return x
                    if isinstance(x, list):
                        return [to_jsonable(i) for i in x]
                    if isinstance(x, dict):
                        return {str(k): to_jsonable(v) for k, v in x.items()}
                    # Fallback: string cast
                    return str(x)
                quiz = to_jsonable(quiz)
            except Exception as e2:
                print(f"[WORKFLOW] Quiz sanitize failed: {e2}; returning empty quiz")
                quiz = {"questions": []}

        return {
            "success": True,
            "step_type": "practice_quiz",
            "quiz": quiz,
            "content": quiz_notification,  # This will be displayed in chat
            "instructions": f"Practice {topic} with these {len(quiz['questions'])} questions. Take your time and think through each one.",
            "next_action": "submit_answers"
        }
    
    async def _execute_review_results(self, step: Dict) -> Dict[str, Any]:
        """Execute review results step"""
        # Get student profile
        student_profile = db.get_topic_proficiency()
        
        # Analyze results
        weak_topics = []
        strong_topics = []
        
        for topic, data in student_profile.items():
            if isinstance(data, dict):
                accuracy = data.get("accuracy", 0.0)
                if accuracy < 0.6:
                    weak_topics.append(topic)
                elif accuracy >= 0.8:
                    strong_topics.append(topic)
        
        return {
            "success": True,
            "step_type": "review_results",
            "student_profile": student_profile,
            "weak_topics": weak_topics,
            "strong_topics": strong_topics,
            "recommendations": self._generate_recommendations(weak_topics, strong_topics),
            "next_action": "continue_learning"
        }
    
    def _generate_recommendations(self, weak_topics: List[str], strong_topics: List[str]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if weak_topics:
            recommendations.append(f"Focus on strengthening: {', '.join(weak_topics[:3])}")
        
        if strong_topics:
            recommendations.append(f"Great progress in: {', '.join(strong_topics[:2])}")
        
        if not weak_topics and not strong_topics:
            recommendations.append("Keep practicing to maintain your current level")
        
        return recommendations
    
    async def submit_quiz_answers(self, answers: List[str]) -> Dict[str, Any]:
        """Submit and evaluate quiz answers, then trigger plan adaptation"""
        if not self.current_state.current_quiz:
            return {
                "success": False,
                "error": "No active quiz to submit answers for"
            }
        
        quiz = self.current_state.current_quiz
        topic = quiz["topic"]
        concept_id = quiz.get("concept_id")
        
        # Evaluate answers
        eval_result = await self.tutor_evaluator.execute(
            "evaluate_quiz",
            quiz=quiz,
            student_answers=answers
        )
        
        if not eval_result.success:
            return {
                "success": False,
                "error": "Quiz evaluation failed",
                "details": eval_result.reasoning
            }
        
        evaluation = eval_result.data
        score_percentage = evaluation["score_percentage"]
        
        # Update student profile
        profile_result = await self.tutor_evaluator.execute(
            "update_profile",
            topic=topic,
            score_percentage=score_percentage,
            concept_id=concept_id
        )
        
        # Get current step info
        current_step = self.current_state.study_plan[self.current_state.current_step] if self.current_state.current_step < len(self.current_state.study_plan) else None
        
        # Tag unclear segments from question-level segment_hints
        unclear_counts = {}
        for r in evaluation.get("results", []):
            seg = r.get("segment_hint", "general")
            if not r.get("is_correct", False):
                unclear_counts[seg] = unclear_counts.get(seg, 0) + 1
        unclear_segments = [seg for seg, cnt in sorted(unclear_counts.items(), key=lambda x: x[1], reverse=True) if seg != "general"]

        # Save quiz results and unclear segments in session memory
        if self.memory:
            try:
                ev = dict(evaluation)
                ev["unclear_segments"] = unclear_segments
                self.memory.append_quiz_result(topic, ev)
                self.memory.update_context({"last_unclear_segments": unclear_segments, "last_quiz_score": score_percentage, "current_topic": topic})
                
                # Update learning objectives mastery based on quiz performance
                segment_id = current_step.get("segment_id") if current_step else None
                if segment_id:
                    segment_plan = self.memory.get_segment_plan(topic)
                    segment_info = next((s for s in segment_plan if s.get("segment_id") == segment_id), None)
                    
                    if segment_info and "learning_objectives" in segment_info:
                        # Calculate confidence based on quiz score and difficulty rating
                        confidence_score = score_percentage / 100.0  # Convert to 0-1
                        
                        for objective in segment_info["learning_objectives"]:
                            self.memory.mark_objective_mastered(topic, objective, confidence_score)
                            print(f"[WORKFLOW] Marked objective: '{objective[:50]}...' with confidence {confidence_score:.2f}")
                
                # Schedule spaced repetition based on quiz performance
                mastery_score = score_percentage / 100.0  # Convert to 0-1
                self.memory.schedule_review(topic, mastery_score)
                next_review = self.memory.get_next_review_time(topic)
                if next_review:
                    print(f"[WORKFLOW] Scheduled review for '{topic}' at {next_review.strftime('%Y-%m-%d %H:%M')}")
                
            except Exception as e:
                print(f"[WORKFLOW] Error updating objectives/spaced repetition: {e}")
                pass

        # Evaluate progress and adapt plan using LLM planner
        print("🔄 Evaluating progress and adapting plan...")
        adaptation_result = await self.planner_agent.evaluate_and_adapt(
            completed_step=current_step or {"topic": topic, "action": "quiz"},
            quiz_results=evaluation,
            student_questions=[]  # Could track questions asked during session
        )
        
        # Log quiz results in lesson history
        if self.current_state.doc_id:
            session_id = self.current_state.session_id or f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=f"Quiz completed! Your score: {score_percentage:.1f}%. {adaptation_result.reasoning}",
                sources=[]
            )
            
            # Update lesson session with quiz results
            db.update_lesson_session(
                session_id=session_id,
                quiz_results=evaluation,
                final_evaluation=adaptation_result.evaluation
            )
            # Append quiz results to session memory for topic-level analytics
            if self.memory and isinstance(evaluation, dict):
                try:
                    ev = dict(evaluation)
                    ev["topic"] = topic
                    self.memory.append_quiz_result(topic, ev)
                except Exception:
                    pass
        
        # Update state based on adaptation
        if adaptation_result.success:
            self.current_state.study_plan = adaptation_result.plan
            next_action = adaptation_result.next_action
            
            # Move to next step
            self.current_state.current_step += 1
            
            return {
                "success": True,
                "evaluation": evaluation,
                "profile_updated": profile_result.success,
                "next_action": next_action,
                "planner_reasoning": adaptation_result.reasoning,
                "planner_evaluation": adaptation_result.evaluation,
                "adapted_plan": adaptation_result.plan if next_action == "adapt_plan" else None,
                "message": f"Quiz completed! Score: {score_percentage:.1f}%. {adaptation_result.reasoning}"
            }
        else:
            # Move to next step even if adaptation fails
            self.current_state.current_step += 1
            
            return {
                "success": True,
                "evaluation": evaluation,
                "profile_updated": profile_result.success,
                "next_action": "continue_plan",
                "message": f"Quiz completed! Score: {score_percentage:.1f}%. Your profile has been updated."
            }

    async def synthesize_episodic_summary(self) -> Dict[str, Any]:
        """Create a concise episodic summary from recent taught segment summaries and quiz results."""
        try:
            # Gather recent summaries and last quiz results for current topic
            recent_summaries = self.memory.get_last_session_summaries(k=5) if self.memory else []
            topic = self.current_state.current_topic
            recent_quiz = self.memory.get_recent_quiz_results(topic, k=2) if (self.memory and topic) else []

            prompt = f"""
You are summarizing a learning session.
RECENT_SUMMARIES:
{json.dumps(recent_summaries)[:1500]}

RECENT_QUIZ_RESULTS:
{json.dumps(recent_quiz)[:800]}

Write a 3-5 sentence episodic summary capturing what was taught and current mastery.
Keep it concise and student-friendly. No markdown, plain text only.
"""
            import asyncio
            response = await asyncio.wait_for(self.llm.ainvoke(prompt), timeout=30.0)
            text = response.content.strip()
            if self.memory:
                self.memory.set_episodic_summary(text, meta={"topic": topic})
            return {"success": True, "summary": text}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def answer_student_question(self, question: str, topic: str = None) -> Dict[str, Any]:
        """Answer student's question using RAG with current topic context"""
        # Use current topic from state if not provided
        if not topic and self.current_state.current_topic:
            topic = self.current_state.current_topic
            print(f"[WORKFLOW] Using current topic from state: {topic}")
        
        # Track engagement: increment questions_asked
        if self.adaptive_state:
            self.adaptive_state["engagement_signals"]["questions_asked"] += 1
        
        # Fetch short-term memory summaries for coherence and current segment taught content
        recent_summaries = []
        recent_qa = []
        current_segment_id = None
        current_segment_text = ""
        if self.memory:
            try:
                recent_summaries = self.memory.get_last_session_summaries(k=3) or []
                recent_qa = self.memory.get_recent_qa(k=3) or []  # NEW: Q&A history
                ctx = self.memory.get_context() or {}
                current_segment_id = ctx.get("current_segment")
                if topic and current_segment_id:
                    current_segment_text = self.memory.get_taught_segment(topic, current_segment_id) or ""
                
                # Log what we're passing for debugging
                print(f"[WORKFLOW] Answer context: topic={topic}, segment={current_segment_id}")
                print(f"[WORKFLOW] Current segment text: {len(current_segment_text)} chars")
                print(f"[WORKFLOW] Recent summaries: {len(recent_summaries)} items")
                print(f"[WORKFLOW] Recent Q&A: {len(recent_qa)} exchanges")
                if recent_summaries:
                    for i, summ in enumerate(recent_summaries[:2]):
                        print(f"  Summary {i+1}: {str(summ)[:80]}...")
                if recent_qa:
                    for i, qa in enumerate(recent_qa[:2]):
                        print(f"  Q&A {i+1}: Q={qa.get('q', '')[:50]}...")
            except Exception as e:
                print(f"[WORKFLOW] Error fetching answer context: {e}")
                recent_summaries = []
                recent_qa = []
        
        answer_result = await self.tutor_evaluator.execute(
            "answer_question",
            question=question,
            topic=topic,
            recent_summaries=recent_summaries,
            recent_qa=recent_qa,  # NEW: Q&A history
            current_segment_id=current_segment_id,
            current_segment_text=current_segment_text
        )
        
        # Store Q&A exchange in memory
        if answer_result.success and self.memory:
            try:
                self.memory.append_qa_exchange(question, answer_result.data["answer"])
                print(f"[WORKFLOW] Stored Q&A exchange in conversation history")
            except Exception as e:
                print(f"[WORKFLOW] Failed to store Q&A: {e}")
        
        if not answer_result.success:
            return {
                "success": False,
                "error": "Answer generation failed",
                "details": answer_result.reasoning
            }
        
        # Log the conversation in lesson history
        if self.current_state.doc_id:
            session_id = f"session_{self.current_state.doc_id}"
            db.add_lesson_message(
                session_id=session_id,
                role="student",
                content=question
            )
            db.add_lesson_message(
                session_id=session_id,
                role="tutor",
                content=answer_result.data["answer"],
                sources=answer_result.data.get("sources", [])
            )
        
        return {
            "success": True,
            "answer": answer_result.data["answer"],
            "sources": answer_result.data.get("sources", []),
            "supporting_chunks": answer_result.data.get("supporting_chunks", [])
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current workflow state"""
        return {
            "doc_id": self.current_state.doc_id,
            "session_id": getattr(self.current_state, 'session_id', None),
            "ingest_status": self.current_state.ingest_status,
            "concepts": self.current_state.concepts,
            "study_plan": self.current_state.study_plan,
            "current_step": self.current_state.current_step,
            "student_choice": self.current_state.student_choice,
            "current_quiz": self.current_state.current_quiz
        }
    
    def get_spaced_repetition_schedule(self) -> Dict[str, Any]:
        """Get spaced repetition schedule for review"""
        student_profile = db.get_topic_proficiency()
        due_topics = self.spaced_rep.get_topics_for_review(student_profile)
        
        return {
            "due_topics": due_topics,
            "total_topics": len(student_profile),
            "next_reviews": {
                topic: self.spaced_rep.calculate_next_review(
                    topic, 
                    data.get("accuracy", 0.0), 
                    data.get("attempts", 0)
                ).isoformat()
                for topic, data in student_profile.items()
                if isinstance(data, dict)
            }
        }

# Global orchestrator instance
workflow_orchestrator = WorkflowOrchestrator()