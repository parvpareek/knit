import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTutor } from '@/context/TutorContext';
import { Sidebar } from '@/components/Sidebar';
import { ChatMessage } from '@/components/ChatMessage';
import { ProgressBar } from '@/components/ProgressBar';
import { LoadingSpinner } from '@/components/LoadingSpinner';
import { Button } from '@/components/ui/button';
import LastSummaries from '@/components/LastSummaries';
import EpisodicSummary from '@/components/EpisodicSummary';
import { Input } from '@/components/ui/input';
import { executeStep, askQuestion, submitExercise, submitDifficultyRating } from '@/api/tutorApi';
import { Send, FileCheck, ArrowRight, Frown, Meh, Smile, Laugh, ThumbsUp } from 'lucide-react';

export default function Learn() {
  const navigate = useNavigate();
  const { state, addMessage, setCurrentQuiz, setCurrentStep, setQuizPending, setCurrentTopic, setPlannerReason, setCurrentSegment } = useTutor();
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [exercises, setExercises] = useState<string[]>([]);
  const [followUps, setFollowUps] = useState<string[]>([]);
  const [exerciseAnswers, setExerciseAnswers] = useState<Record<string, string>>({});
  const [exerciseSubmitted, setExerciseSubmitted] = useState(false);
  const [needsRating, setNeedsRating] = useState(false);
  const [ratingContext, setRatingContext] = useState<{topic: string; segment_id: string} | null>(null);
  const [submittedRating, setSubmittedRating] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (state.sessionId && state.conversationHistory.length === 0) {
      loadInitialContent();
    } else if (state.conversationHistory.length > 0) {
      // Already have content, not initial load
      setInitialLoading(false);
    }
  }, [state.sessionId, state.conversationHistory.length]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [state.conversationHistory]);

  const loadInitialContent = async () => {
    console.log('[Learn] Loading initial content for step 0...');
    try {
      const response = await executeStep(0);
      console.log('[Learn] Initial content response:', response);
      
      // Update topic/segment from response
      if (response.topic) {
        console.log('[Learn] Updating topic to:', response.topic);
        setCurrentTopic(response.topic);
      }
      if (response.segment_id || response.segment_title) {
        setCurrentSegment(response.segment_id || null, response.segment_title || null);
      }
      if (response.planner_reason) {
        setPlannerReason(response.planner_reason);
      }
      
      if (response.content) {
        addMessage({ role: 'tutor', content: response.content });
      } else {
        console.warn('[Learn] No content in response:', response);
        addMessage({ role: 'tutor', content: 'Ready to start learning! Ask me any questions about the uploaded material.' });
      }
      setExercises(response.exercises || []);
      setFollowUps(response.follow_ups || []);
      
      // Check if rating is needed
      if (response.request_difficulty_rating && response.rating_context) {
        setNeedsRating(true);
        setRatingContext(response.rating_context);
        setSubmittedRating(false);
      }
      
      // Store quiz if this is a quiz step
      if (response.quiz) {
        setCurrentQuiz(response.quiz);
      }
    } catch (error) {
      console.error('[Learn] Failed to load initial content:', error);
      addMessage({ role: 'tutor', content: 'Failed to load content. Please try asking a question or click Next Step.' });
    } finally {
      setInitialLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!input.trim() || loading) return;

    const question = input;
    setInput('');
    addMessage({ role: 'student', content: question });
    setLoading(true);

    try {
      const response = await askQuestion(question, state.currentTopic);
      addMessage({ 
        role: 'tutor', 
        content: response.answer || 'No answer provided',
        sources: response.sources 
      });
    } catch (error) {
      addMessage({ role: 'tutor', content: 'Failed to get answer. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  const handleTakeQuiz = () => {
    // Simply navigate to quiz if one is available
    if (state.currentQuiz) {
      navigate('/quiz');
    } else {
      addMessage({ role: 'tutor', content: 'No quiz available yet. Please complete the current study topic first.' });
    }
  };

  const handleRatingSubmit = async (rating: number) => {
    if (!ratingContext) return;
    
    try {
      await submitDifficultyRating(
        ratingContext.topic, 
        ratingContext.segment_id, 
        rating
      );
      setSubmittedRating(true);
      setNeedsRating(false);
      console.log(`[Learn] Submitted rating ${rating} for ${ratingContext.topic}`);
    } catch (error) {
      console.error('[Learn] Failed to submit rating:', error);
      // Don't block user, just log error
    }
  };

  const handleNextStep = async () => {
    // Block if quiz is pending
    if (state.quizPending) {
      addMessage({ 
        role: 'tutor', 
        content: '⚠️ Please complete the quiz before moving to the next topic. Click "Take Quiz" button to begin!' 
      });
      return;
    }

    setLoading(true);
    try {
      const nextStep = state.currentStep + 1;
      const response = await executeStep(nextStep);
      
      // STEP 1 FIX: Update step AFTER successful response
      setCurrentStep(nextStep);
      
      // Update topic/segment from response
      if (response.topic) {
        console.log('[Learn] Updating topic to:', response.topic);
        setCurrentTopic(response.topic);
      }
      if (response.segment_id || response.segment_title) {
        setCurrentSegment(response.segment_id || null, response.segment_title || null);
      }
      
      // Display content if available
      if (response.content) {
        addMessage({ role: 'tutor', content: response.content });
      }
      if (response.planner_reasoning) {
        setPlannerReason(response.planner_reasoning || response.planner_reason || '');
      }
      setExercises(response.exercises || []);
      setFollowUps(response.follow_ups || []);
      
      // Check if rating is needed
      if (response.request_difficulty_rating && response.rating_context) {
        setNeedsRating(true);
        setRatingContext(response.rating_context);
        setSubmittedRating(false);
      }
      
      // Store quiz if available
      if (response.quiz) {
        setCurrentQuiz(response.quiz);
      }
    } catch (error) {
      console.error('Next step error:', error);
      addMessage({ role: 'tutor', content: 'Failed to load next step. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  if (!state.sessionId) {
    navigate('/');
    return null;
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col bg-background">
        <header className="bg-card border-b border-border p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-baseline gap-2">
                <h2 className="text-xl font-semibold text-card-foreground">{state.currentTopic}</h2>
                {(state.currentSegmentTitle || state.currentSegmentId) && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded bg-primary/10 text-primary text-xs font-medium">
                    {state.currentSegmentTitle || state.currentSegmentId}
                  </span>
                )}
              </div>
            </div>
            <ProgressBar 
              completed={state.progress.completed}
              current={state.currentStep}
              total={state.progress.total}
            />
            {state.plannerReason && (
              <div className="mt-2 text-sm text-muted-foreground">
                <span className="font-medium">Why next:</span> {state.plannerReason}
              </div>
            )}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto p-4 space-y-6">
            <LastSummaries />
            <EpisodicSummary />
            {initialLoading ? (
              <LoadingSpinner message="Loading study content..." />
            ) : (
              state.conversationHistory.map((msg, idx) => (
                <ChatMessage key={idx} {...msg} />
              ))
            )}
            {(exercises.length > 0 || followUps.length > 0) && (
              <div className="bg-card border border-border rounded p-4">
                <div className="text-sm font-semibold mb-2">Exercises & Follow-ups</div>
                {[...new Set([...exercises, ...followUps])].map((q, i) => (
                  <div key={i} className="mb-3">
                    <div className="text-sm mb-1">{q}</div>
                    <Input
                      value={exerciseAnswers[q] || ''}
                      onChange={(e) => setExerciseAnswers(prev => ({ ...prev, [q]: e.target.value }))}
                      placeholder="Type your answer (optional)"
                    />
                    <div className="mt-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={async () => {
                          try {
                            if (exerciseAnswers[q]) {
                              await submitExercise({
                                topic: state.currentTopic,
                                segment_id: state.currentSegmentId || 'unknown',
                                prompt: q,
                                student_answer: exerciseAnswers[q],
                              });
                              setExerciseSubmitted(true);
                              addMessage({ role: 'tutor', content: 'Got it! Logged your exercise response.' });
                            } else {
                              addMessage({ role: 'tutor', content: 'Answer is empty; nothing recorded.' });
                            }
                          } catch {
                            addMessage({ role: 'tutor', content: 'Failed to record exercise response.' });
                          }
                        }}
                      >
                        Submit
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {loading && <LoadingSpinner message="Thinking..." size="sm" />}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="bg-card border-t border-border p-4">
          <div className="max-w-4xl mx-auto space-y-3">
            {/* Difficulty Rating Buttons */}
            {needsRating && !submittedRating && (
              <div className="bg-muted/50 border border-border rounded-lg p-4">
                <div className="text-sm font-medium mb-3 text-center">
                  How comfortable did you feel with this material?
                </div>
                <div className="flex gap-2 justify-center">
                  <Button
                    onClick={() => handleRatingSubmit(1)}
                    variant="outline"
                    size="lg"
                    className="flex-col h-auto py-3 px-4 hover:bg-red-100 hover:border-red-300"
                    title="Too hard / Confused"
                  >
                    <Frown className="h-6 w-6 mb-1 text-red-500" />
                    <span className="text-xs">Too Hard</span>
                  </Button>
                  <Button
                    onClick={() => handleRatingSubmit(2)}
                    variant="outline"
                    size="lg"
                    className="flex-col h-auto py-3 px-4 hover:bg-orange-100 hover:border-orange-300"
                    title="Challenging"
                  >
                    <Meh className="h-6 w-6 mb-1 text-orange-500" />
                    <span className="text-xs">Challenging</span>
                  </Button>
                  <Button
                    onClick={() => handleRatingSubmit(3)}
                    variant="outline"
                    size="lg"
                    className="flex-col h-auto py-3 px-4 hover:bg-blue-100 hover:border-blue-300"
                    title="Just right"
                  >
                    <Smile className="h-6 w-6 mb-1 text-blue-500" />
                    <span className="text-xs">Just Right</span>
                  </Button>
                  <Button
                    onClick={() => handleRatingSubmit(4)}
                    variant="outline"
                    size="lg"
                    className="flex-col h-auto py-3 px-4 hover:bg-green-100 hover:border-green-300"
                    title="Easy"
                  >
                    <Laugh className="h-6 w-6 mb-1 text-green-500" />
                    <span className="text-xs">Easy</span>
                  </Button>
                  <Button
                    onClick={() => handleRatingSubmit(5)}
                    variant="outline"
                    size="lg"
                    className="flex-col h-auto py-3 px-4 hover:bg-emerald-100 hover:border-emerald-300"
                    title="Too easy / Boring"
                  >
                    <ThumbsUp className="h-6 w-6 mb-1 text-emerald-500" />
                    <span className="text-xs">Too Easy</span>
                  </Button>
                </div>
              </div>
            )}

            <div className="flex gap-2">
              {state.currentQuiz && (
                <Button 
                  onClick={handleTakeQuiz}
                  variant={state.quizPending ? "default" : "outline"}
                  disabled={loading}
                  className={state.quizPending ? "animate-pulse" : ""}
                >
                  <FileCheck className="mr-2 h-4 w-4" />
                  {state.quizPending ? "Complete Quiz to Continue" : "Take Quiz"}
                </Button>
              )}
              <Button 
                onClick={handleNextStep}
                variant="outline"
                disabled={loading || state.quizPending}
                title={state.quizPending ? "Complete the quiz first" : ""}
              >
                <ArrowRight className="mr-2 h-4 w-4" />
                Next Step
              </Button>
            </div>

            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleAskQuestion()}
                placeholder="Ask a question..."
                disabled={loading}
              />
              <Button 
                onClick={handleAskQuestion}
                disabled={loading || !input.trim()}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
