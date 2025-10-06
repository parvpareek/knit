import { useNavigate } from 'react-router-dom';
import { useTutor } from '@/context/TutorContext';
import { ConceptList } from '@/components/ConceptList';
import { Button } from '@/components/ui/button';
import { BookOpen, ArrowRight } from 'lucide-react';
import { SegmentList } from '@/components/SegmentList';

export default function Concepts() {
  const navigate = useNavigate();
  const { state } = useTutor();

  if (state.concepts.length === 0) {
    navigate('/');
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary rounded-2xl mb-4">
            <BookOpen className="h-8 w-8 text-primary-foreground" />
          </div>
          <h1 className="text-4xl font-bold text-foreground mb-3">
            Found {state.concepts.length} Key Concepts
          </h1>
          <p className="text-lg text-muted-foreground">
            Your personalized study plan is ready. Let's master these topics!
          </p>
        </div>

        <div className="bg-card rounded-xl p-8 border border-border shadow-sm mb-8">
          <ConceptList 
            concepts={state.concepts}
            completedCount={state.progress.completed}
          />
        </div>

        <div className="bg-card rounded-xl p-8 border border-border shadow-sm mb-8">
          <h2 className="text-lg font-semibold text-card-foreground mb-3">Segments by Topic</h2>
          <SegmentList concepts={state.concepts as any} />
        </div>

        <div className="bg-primary/5 rounded-xl p-6 border border-primary/20 mb-8">
          <p className="text-card-foreground font-medium mb-1">Starting with:</p>
          <p className="text-2xl font-bold text-primary">{state.concepts[0].label}</p>
        </div>

        <Button 
          onClick={() => navigate('/learn')}
          size="lg"
          className="w-full group"
        >
          Begin Learning
          <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
        </Button>
      </div>
    </div>
  );
}
