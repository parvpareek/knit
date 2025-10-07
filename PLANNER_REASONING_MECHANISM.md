# 🧠 Planner Reasoning Mechanism - Transparent AI Decisions

## Overview
Implemented a comprehensive mechanism to capture, log, and display the **planner's reasoning** behind every decision, making the adaptive learning system fully transparent to students.

---

## 🎯 What Was Added

### 1. LLM Decision Reasoning Capture (`llm_planner.py`)

**Enhanced `_decide_next_action_llm()`:**
- Now returns tuple `(action, reasoning)` instead of just action
- Reasoning extracted from LLM response: `"ACTION: reasoning"`
- Example LLM response: `"move_forward: Strong performance indicates readiness for next concept"`

**Added Fallback Reasoning:**
- New method `_get_fallback_reasoning()` generates explanatory text for rule-based decisions
- Examples:
  - `"Score of 85% - ready to advance to next concept"`
  - `"Score of 45% indicates need for clarification on fundamentals"`
  - `"Moderate score of 65% - adding targeted practice"`

**Reasoning Storage:**
- Added `self.last_decision_reason` instance variable
- Captured in `_decide_next_action()` wrapper
- Used in all `PlannerResponse` objects

### 2. Context-Rich Decision Making

The LLM receives comprehensive context to make informed decisions:

```python
**Context Provided to LLM:**
1. Quiz Performance
   - Score percentage
   - Unclear segments
   - Previous attempts

2. Learning Patterns
   - Confusion type (concept/application/connection)
   - Is this a repeat question?
   - Suggested teaching approach

3. Engagement Profile
   - Engagement level (high/medium/low)
   - Preferred learning style (visual/conceptual/practical)
   - Student needs (examples, connections, practice)

4. Mastery Status
   - Unmastered objectives count
   - Recent Q&A activity

5. Evaluation Insights
   - Needs clarification flag
   - Ready for next topic flag
```

**Example Decision Prompt:**
```
You are an adaptive learning planner. Based on student performance, decide the BEST next action.

**Quiz Results:** Text Processing Basics - 65% score
  - Unclear segments: tokenization, normalization
  - Previous attempts: 1

**Confusion Pattern:** concept (needs foundational clarity)

**Engagement:** high engagement, conceptual learner
  - Needs: more examples; clearer connections

**Mastery:** 2 objectives not yet mastered

Consider:
- Low scores (<50%) → usually need clarification
- Moderate scores (50-80%) with confusion patterns → adapt teaching approach
- High scores (>80%) with mastery → move forward

Respond with ONLY the action name and brief 1-sentence reasoning.
Format: ACTION: reasoning
```

### 3. Reasoning Flow Through System

**Backend Flow:**
1. **Planner makes decision** → Captures reasoning in `self.last_decision_reason`
2. **PlannerResponse created** → Includes `reasoning` field from `self.last_decision_reason`
3. **Orchestrator receives response** → Extracts `adaptation_result.reasoning`
4. **API response includes reasoning** → Both `planner_reasoning` and `planner_reason` fields

**API Response Structure:**
```python
{
    "success": True,
    "planner_reasoning": "LLM decision reasoning here",
    "planner_reason": "LLM decision reasoning here",  # For frontend compatibility
    "planner_evaluation": {...},
    "next_action": "move_forward",
    "message": "Quiz completed! Score: 85%. LLM decision reasoning here"
}
```

### 4. Reasoning for All Step Types

**Study Segments:**
```python
{
    "step_type": "study_segment",
    "planner_reason": "Learn this segment",  # From step.why_assigned
    ...
}
```

**Quizzes:**
```python
{
    "success": True,
    "planner_reasoning": "Strong score of 85% - ready to advance",
    "planner_reason": "Strong score of 85% - ready to advance",
    ...
}
```

**Optional Exercises:**
```python
{
    "step_type": "optional_exercise",
    "planner_reason": "Optional exercise to reinforce understanding",
    ...
}
```

### 5. Frontend Display (`Learn.tsx`)

**Already implemented UI:**
```tsx
{state.plannerReason && (
  <div className="mt-2 text-sm text-muted-foreground">
    <span className="font-medium">Why next:</span> {state.plannerReason}
  </div>
)}
```

**Visual Example:**
```
Topic: Text Processing Basics             Step 5 of 12
Progress: [==================          ] 42% complete
Why next: Strong quiz performance - advancing to next concept

[Content appears here...]
```

---

## 🔍 Example Scenarios

### Scenario 1: Low Quiz Score
**Input:**
- Quiz score: 45%
- Unclear segments: tokenization
- Confusion pattern: concept fundamentals

**LLM Decision:**
```
Action: clarify_concept
Reasoning: Low score indicates fundamental gaps in understanding tokenization - needs clarification with different approach
```

**Frontend Display:**
```
Why next: Low score indicates fundamental gaps in understanding tokenization - needs clarification with different approach
```

---

### Scenario 2: Moderate Score with High Engagement
**Input:**
- Quiz score: 68%
- Recent questions: 5
- Confusion pattern: application (needs practical examples)

**LLM Decision:**
```
Action: adapt_plan
Reasoning: Moderate performance but high engagement suggests student wants to understand deeply - adding practical examples
```

**Frontend Display:**
```
Why next: Moderate performance but high engagement suggests student wants to understand deeply - adding practical examples
```

---

### Scenario 3: High Score, Ready to Advance
**Input:**
- Quiz score: 92%
- Mastery: all objectives mastered
- Engagement: low (no questions)

**LLM Decision:**
```
Action: move_forward
Reasoning: Excellent score with full mastery - student ready for next concept
```

**Frontend Display:**
```
Why next: Excellent score with full mastery - student ready for next concept
```

---

## 📊 Types of Reasoning

### 1. **Quiz-Based Decisions**
- Performance evaluation
- Segment understanding analysis
- Adaptive difficulty adjustments

### 2. **Engagement-Based Decisions**
- Question patterns analysis
- Learning style adaptation
- Interest level recognition

### 3. **Mastery-Based Decisions**
- Objective completion tracking
- Concept readiness assessment
- Progression gating

### 4. **Confusion Pattern Decisions**
- Repeat question detection
- Confusion type identification
- Teaching strategy adaptation

---

## 🛠️ Technical Implementation

### Files Modified

1. **`backend/app/agents/llm_planner.py`**
   - Modified `_decide_next_action_llm()` to return `(action, reasoning)` tuple
   - Added `_get_fallback_reasoning()` for rule-based decisions
   - Added `self.last_decision_reason` instance variable
   - Updated all `PlannerResponse` objects to use `self.last_decision_reason`

2. **`backend/app/agents/workflow_orchestrator.py`**
   - Added `planner_reason` to study segment responses (from `step.why_assigned`)
   - Added `planner_reason` to quiz responses (from `adaptation_result.reasoning`)
   - Added `planner_reason` to optional exercise responses

3. **`frontend/src/pages/Learn.tsx`**
   - Already configured to display `state.plannerReason`
   - Checks for both `planner_reason` and `planner_reasoning` fields

---

## 🎨 UI/UX Benefits

### For Students:
- **Transparency** - Understand why the system makes decisions
- **Trust** - See the reasoning behind adaptive changes
- **Learning** - Gain insight into their own progress patterns

### For Educators/Admins:
- **Debuggability** - Track decision-making logic
- **Accountability** - Verify adaptive behavior is sound
- **Insights** - Understand how students progress

---

## 🔮 Example Logs

### Console Output:
```
[LLMPlannerAgent] 🧠 LLM Decision: move_forward - Strong score of 85% with all objectives mastered indicates readiness

[WORKFLOW] Quiz completed | Score: 85% | Action: move_forward
[WORKFLOW] Planner reasoning: Strong score of 85% with all objectives mastered indicates readiness

[API] Returning planner_reason: Strong score of 85% with all objectives mastered indicates readiness
```

### Frontend Display:
```
┌─────────────────────────────────────────────┐
│ Text Processing Basics                      │
│ Step 6 of 12                  50% complete  │
│ ──────────────────────────────              │
│                                              │
│ Why next: Strong score of 85% with all      │
│ objectives mastered indicates readiness     │
└─────────────────────────────────────────────┘
```

---

## ✅ Testing Checklist

- [x] LLM decision reasoning captured correctly
- [x] Fallback reasoning generated for rule-based decisions
- [x] `PlannerResponse` includes reasoning
- [x] Study segment responses include `planner_reason`
- [x] Quiz responses include both `planner_reasoning` and `planner_reason`
- [x] Optional exercise responses include `planner_reason`
- [x] Frontend displays reasoning in UI
- [x] No linter errors
- [x] Backwards compatible with existing frontend

---

## 🚀 Impact

### Before:
```
[System makes decision]
→ Student sees: "Next Step" button
→ No explanation
```

### After:
```
[System makes decision with rich context]
→ LLM evaluates: quiz scores, confusion patterns, engagement, mastery
→ Decision: "move_forward"
→ Reasoning: "Strong performance with deep engagement - ready to advance"
→ Student sees: 
   "Why next: Strong performance with deep engagement - ready to advance"
→ Transparency ✅
```

---

**Status: ✅ Complete**  
**Impact: Full transparency in adaptive decision-making**  
**User Experience: Students understand why the system adapts to them**

