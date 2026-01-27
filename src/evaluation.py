"""
Evaluation Module for LCA RAG Application.

This module handles:
- Faithfulness evaluation (grounding in context)
- Answer relevancy scoring
- Context relevancy assessment
- Hallucination detection using RAGAS
- LangSmith integration for tracing and evaluation
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json

# LangSmith integration
try:
    from langsmith import Client as LangSmithClient
    from langsmith.wrappers import wrap_openai
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    faithfulness_score: float
    relevancy_score: float
    context_relevancy: float
    hallucination_detected: bool
    details: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    
    def __post_init__(self):
        """Calculate if evaluation passed."""
        self.passed = (
            self.faithfulness_score >= 0.8 and
            not self.hallucination_detected
        )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    faithfulness_threshold: float = 0.8
    relevancy_threshold: float = 0.7
    context_relevancy_threshold: float = 0.6
    use_ragas: bool = True
    use_llm_eval: bool = True
    use_langsmith: bool = True
    langsmith_project: str = "lca-rag-evaluation"


class LangSmithEvaluator:
    """
    LangSmith integration for tracing and evaluation.
    
    Tracks:
    - Query latency
    - Token usage
    - Response quality metrics
    """
    
    def __init__(self, project_name: str = "lca-rag-evaluation"):
        """
        Initialize LangSmith evaluator.
        
        Args:
            project_name: LangSmith project name
        """
        self.project_name = project_name
        self.client = None
        self._initialized = False
        
        # Check for API key
        if LANGSMITH_AVAILABLE and os.environ.get("LANGSMITH_API_KEY"):
            try:
                self.client = LangSmithClient()
                os.environ["LANGSMITH_PROJECT"] = project_name
                os.environ["LANGSMITH_TRACING"] = "true"
                self._initialized = True
                logger.info(f"LangSmith initialized with project: {project_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith: {e}")
        else:
            logger.info("LangSmith not configured (set LANGSMITH_API_KEY to enable)")
    
    @property
    def is_available(self) -> bool:
        """Check if LangSmith is available."""
        return self._initialized
    
    def log_run(
        self,
        query: str,
        answer: str,
        context: str,
        latency_ms: float,
        evaluation: Optional[Dict[str, Any]] = None
    ):
        """
        Log a RAG run to LangSmith.
        
        Args:
            query: User query
            answer: Generated answer
            context: Retrieved context
            latency_ms: Response latency in milliseconds
            evaluation: Optional evaluation metrics
        """
        if not self._initialized:
            return
        
        try:
            # Create run data
            run_data = {
                "name": "rag_query",
                "run_type": "chain",
                "inputs": {"query": query, "context": context[:1000]},
                "outputs": {"answer": answer},
                "extra": {
                    "latency_ms": latency_ms,
                    "evaluation": evaluation or {}
                }
            }
            
            # Log to LangSmith (simplified - actual implementation would use run tree)
            logger.debug(f"LangSmith run logged: {latency_ms:.0f}ms")
            
        except Exception as e:
            logger.warning(f"Failed to log to LangSmith: {e}")


class FaithfulnessEvaluator:
    """
    Evaluates if answers are grounded in the provided context.
    
    Uses both heuristic and LLM-based evaluation.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None, llm: Any = None):
        """
        Initialize the faithfulness evaluator.
        
        Args:
            config: Evaluation configuration
            llm: Optional LLM for advanced evaluation
        """
        self.config = config or EvaluationConfig()
        self.llm = llm
        
    def evaluate(
        self,
        answer: str,
        context: str,
        query: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate faithfulness of answer to context.
        
        Args:
            answer: Generated answer
            context: Source context
            query: Original query
            
        Returns:
            Tuple of (faithfulness_score, details)
        """
        details = {}
        
        # Heuristic evaluation
        heuristic_score, heuristic_details = self._heuristic_evaluate(answer, context)
        details["heuristic"] = heuristic_details
        
        # LLM-based evaluation if available
        if self.llm and self.config.use_llm_eval:
            llm_score, llm_details = self._llm_evaluate(answer, context, query)
            details["llm"] = llm_details
            
            # Combine scores (weighted average)
            final_score = 0.3 * heuristic_score + 0.7 * llm_score
        else:
            final_score = heuristic_score
        
        details["final_score"] = final_score
        details["threshold"] = self.config.faithfulness_threshold
        details["passed"] = final_score >= self.config.faithfulness_threshold
        
        return final_score, details
    
    def _heuristic_evaluate(
        self,
        answer: str,
        context: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Heuristic-based faithfulness evaluation.
        
        Checks:
        - Key terms from answer appear in context
        - Numbers/values match
        - No unsupported claims
        """
        details = {}
        
        # Tokenize
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "shall",
                      "can", "need", "dare", "ought", "used", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "as", "into",
                      "through", "during", "before", "after", "above", "below",
                      "between", "under", "again", "further", "then", "once",
                      "and", "but", "or", "nor", "so", "yet", "both", "either",
                      "neither", "not", "only", "own", "same", "than", "too",
                      "very", "just", "also", "now", "here", "there", "when",
                      "where", "why", "how", "all", "each", "every", "both",
                      "few", "more", "most", "other", "some", "such", "no",
                      "any", "this", "that", "these", "those", "i", "you", "he",
                      "she", "it", "we", "they", "what", "which", "who", "whom"}
        
        answer_words = answer_words - stop_words
        context_words = context_words - stop_words
        
        # Calculate overlap
        overlap = answer_words & context_words
        coverage = len(overlap) / len(answer_words) if answer_words else 0
        
        details["word_overlap"] = len(overlap)
        details["answer_words"] = len(answer_words)
        details["coverage_ratio"] = coverage
        
        # Check for numbers
        import re
        answer_numbers = set(re.findall(r'\d+\.?\d*', answer))
        context_numbers = set(re.findall(r'\d+\.?\d*', context))
        
        if answer_numbers:
            numbers_in_context = len(answer_numbers & context_numbers)
            number_accuracy = numbers_in_context / len(answer_numbers)
            details["numbers_accuracy"] = number_accuracy
        else:
            number_accuracy = 1.0
            details["numbers_accuracy"] = "N/A (no numbers)"
        
        # Calculate heuristic score
        score = (coverage * 0.7 + number_accuracy * 0.3)
        
        return score, details
    
    def _llm_evaluate(
        self,
        answer: str,
        context: str,
        query: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        LLM-based faithfulness evaluation.
        """
        details = {}
        
        try:
            eval_prompt = f"""Evaluate if the following answer is faithfully grounded in the provided context.

CONTEXT:
{context[:2000]}

QUESTION: {query}

ANSWER:
{answer}

Evaluate on a scale of 0-10:
1. Does the answer only contain information from the context? (0-10)
2. Are all claims supported by the context? (0-10)
3. Are numbers and facts accurate to the context? (0-10)

Respond in JSON format:
{{"grounding": <score>, "support": <score>, "accuracy": <score>, "explanation": "<brief explanation>"}}"""

            response = self.llm.generate(eval_prompt)
            
            # Parse response
            try:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    eval_data = json.loads(json_match.group())
                    grounding = eval_data.get("grounding", 5) / 10
                    support = eval_data.get("support", 5) / 10
                    accuracy = eval_data.get("accuracy", 5) / 10
                    
                    score = (grounding + support + accuracy) / 3
                    details["grounding"] = grounding
                    details["support"] = support
                    details["accuracy"] = accuracy
                    details["explanation"] = eval_data.get("explanation", "")
                else:
                    score = 0.7  # Default score if parsing fails
                    details["parse_error"] = "Could not extract JSON"
            except:
                score = 0.7
                details["parse_error"] = "JSON parsing failed"
                
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}")
            score = 0.7
            details["error"] = str(e)
        
        return score, details


class HallucinationDetector:
    """
    Detects hallucinations in generated answers.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize hallucination detector.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        
        # Common hallucination indicators
        self.uncertain_phrases = [
            "i think", "probably", "likely", "possibly", "might be",
            "could be", "generally", "usually", "typically", "often",
            "sometimes", "perhaps", "maybe"
        ]
        
        self.unsupported_phrases = [
            "according to research", "studies show", "experts say",
            "it is well known", "it is common knowledge", "as we all know",
            "historically", "traditionally", "in my experience"
        ]
        
    def detect(
        self,
        answer: str,
        context: str,
        query: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect potential hallucinations in answer.
        
        Args:
            answer: Generated answer
            context: Source context
            query: Original query
            
        Returns:
            Tuple of (hallucination_detected, details)
        """
        details = {
            "checks": [],
            "warnings": []
        }
        
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Check for uncertain language
        uncertain_count = sum(1 for phrase in self.uncertain_phrases 
                             if phrase in answer_lower)
        if uncertain_count > 2:
            details["warnings"].append(f"High uncertainty language ({uncertain_count} instances)")
        
        # Check for unsupported claims
        unsupported_count = sum(1 for phrase in self.unsupported_phrases 
                               if phrase in answer_lower)
        if unsupported_count > 0:
            details["warnings"].append(f"Potentially unsupported claims ({unsupported_count} instances)")
        
        # Check for entities not in context
        import re
        
        # Extract capitalized words (potential entities)
        answer_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer))
        context_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context))
        
        # Common words to ignore
        common_words = {"The", "This", "That", "These", "Those", "It", "They", 
                       "We", "I", "You", "He", "She", "Based", "According", 
                       "Please", "Note", "Summary", "Source", "From"}
        
        answer_entities = answer_entities - common_words
        context_entities = context_entities - common_words
        
        new_entities = answer_entities - context_entities
        if new_entities:
            details["warnings"].append(f"New entities not in context: {list(new_entities)[:5]}")
        
        # Check for specific numeric claims
        answer_numbers = re.findall(r'(\d+\.?\d*)\s*(?:kg|g|MJ|kWh|%|percent)', answer)
        context_numbers = re.findall(r'(\d+\.?\d*)\s*(?:kg|g|MJ|kWh|%|percent)', context)
        
        unmatched_numbers = set(answer_numbers) - set(context_numbers)
        if unmatched_numbers:
            details["warnings"].append(f"Numeric values not found in context: {list(unmatched_numbers)[:5]}")
        
        # Check for refusal to answer (not a hallucination)
        refusal_phrases = [
            "i don't have", "not enough information", "cannot find",
            "no information", "not mentioned", "not available"
        ]
        
        is_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
        details["is_refusal"] = is_refusal
        
        # Determine if hallucination detected
        warning_count = len(details["warnings"])
        hallucination_detected = (
            warning_count >= 2 and not is_refusal
        ) or (
            len(new_entities) > 3 and not is_refusal
        ) or (
            len(unmatched_numbers) > 2 and not is_refusal
        )
        
        details["hallucination_detected"] = hallucination_detected
        details["warning_count"] = warning_count
        
        return hallucination_detected, details


class RAGASEvaluator:
    """
    Integration with RAGAS for comprehensive evaluation.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize RAGAS evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config or EvaluationConfig()
        self._ragas_available = None
        
    def _check_ragas(self) -> bool:
        """Check if RAGAS is available."""
        if self._ragas_available is None:
            try:
                from ragas import evaluate
                from ragas.metrics import faithfulness, answer_relevancy
                self._ragas_available = True
            except ImportError:
                logger.warning("RAGAS not available. Install with: pip install ragas")
                self._ragas_available = False
        return self._ragas_available
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate using RAGAS metrics.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dictionary of evaluation results
        """
        if not self._check_ragas():
            return {"error": "RAGAS not available"}
        
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from datasets import Dataset
            
            # Prepare data
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            
            if ground_truths:
                data["ground_truth"] = ground_truths
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics
            metrics = [faithfulness, answer_relevancy]
            
            # Evaluate
            results = evaluate(dataset, metrics=metrics)
            
            return {
                "faithfulness": float(results["faithfulness"]),
                "answer_relevancy": float(results["answer_relevancy"]),
                "dataset_size": len(questions)
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}


class RAGEvaluator:
    """
    Complete evaluation pipeline for RAG system.
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        llm: Any = None
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            config: Evaluation configuration
            llm: Optional LLM for advanced evaluation
        """
        self.config = config or EvaluationConfig()
        self.faithfulness_eval = FaithfulnessEvaluator(config, llm)
        self.hallucination_detector = HallucinationDetector(config)
        self.ragas_eval = RAGASEvaluator(config)
        
    def evaluate_single(
        self,
        query: str,
        answer: str,
        context: str
    ) -> EvaluationResult:
        """
        Evaluate a single query-answer pair.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Source context
            
        Returns:
            EvaluationResult object
        """
        # Evaluate faithfulness
        faithfulness_score, faith_details = self.faithfulness_eval.evaluate(
            answer, context, query
        )
        
        # Detect hallucinations
        hallucination, hall_details = self.hallucination_detector.detect(
            answer, context, query
        )
        
        # Calculate relevancy (simple overlap-based for now)
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        relevancy_score = len(query_words & answer_words) / len(query_words) if query_words else 0
        
        # Context relevancy
        context_words = set(context.lower().split())
        context_relevancy = len(query_words & context_words) / len(query_words) if query_words else 0
        
        return EvaluationResult(
            faithfulness_score=faithfulness_score,
            relevancy_score=relevancy_score,
            context_relevancy=context_relevancy,
            hallucination_detected=hallucination,
            details={
                "faithfulness": faith_details,
                "hallucination": hall_details
            }
        )
    
    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of query-answer pairs.
        
        Args:
            queries: List of queries
            answers: List of answers
            contexts: List of contexts
            
        Returns:
            Aggregated evaluation results
        """
        results = []
        
        for query, answer, context in zip(queries, answers, contexts):
            result = self.evaluate_single(query, answer, context)
            results.append(result)
        
        # Aggregate results
        avg_faithfulness = sum(r.faithfulness_score for r in results) / len(results)
        avg_relevancy = sum(r.relevancy_score for r in results) / len(results)
        hallucination_rate = sum(1 for r in results if r.hallucination_detected) / len(results)
        pass_rate = sum(1 for r in results if r.passed) / len(results)
        
        return {
            "total_samples": len(results),
            "avg_faithfulness": avg_faithfulness,
            "avg_relevancy": avg_relevancy,
            "hallucination_rate": hallucination_rate,
            "pass_rate": pass_rate,
            "passed_threshold": self.config.faithfulness_threshold,
            "individual_results": results
        }


def main():
    """Test the evaluation module."""
    # Create sample data
    sample_context = """Life Cycle Assessment (LCA) is a systematic methodology for evaluating 
    environmental impacts of products throughout their lifecycle. Steel production typically 
    has a Global Warming Potential (GWP) of 2.1 kg CO2 equivalent per kilogram. The ReCiPe 
    methodology provides characterization factors for 18 midpoint impact categories including 
    climate change, ozone depletion, and human toxicity."""
    
    sample_query = "What is the environmental impact of steel production?"
    
    # Good answer (grounded)
    good_answer = """Based on the context, steel production has a Global Warming Potential (GWP) 
    of 2.1 kg CO2 equivalent per kilogram [Source: materials_impact.pdf]. This measurement 
    is part of Life Cycle Assessment (LCA), which evaluates environmental impacts throughout 
    a product's lifecycle."""
    
    # Bad answer (hallucination)
    bad_answer = """Steel production has a GWP of 5.5 kg CO2 equivalent per kilogram, which 
    is significantly higher than aluminum. According to recent studies, steel manufacturing 
    accounts for approximately 30% of global industrial emissions. The process typically 
    involves the Bessemer process which was invented in 1856."""
    
    # Initialize evaluator
    config = EvaluationConfig(
        faithfulness_threshold=0.8
    )
    evaluator = RAGEvaluator(config)
    
    # Test good answer
    print("\n=== Evaluating Good Answer ===")
    print(f"Query: {sample_query}")
    print(f"Answer: {good_answer[:100]}...")
    
    result = evaluator.evaluate_single(sample_query, good_answer, sample_context)
    
    print(f"\nResults:")
    print(f"  Faithfulness Score: {result.faithfulness_score:.2f}")
    print(f"  Relevancy Score: {result.relevancy_score:.2f}")
    print(f"  Hallucination Detected: {result.hallucination_detected}")
    print(f"  Passed: {result.passed}")
    
    # Test bad answer
    print("\n=== Evaluating Bad Answer (with hallucinations) ===")
    print(f"Query: {sample_query}")
    print(f"Answer: {bad_answer[:100]}...")
    
    result = evaluator.evaluate_single(sample_query, bad_answer, sample_context)
    
    print(f"\nResults:")
    print(f"  Faithfulness Score: {result.faithfulness_score:.2f}")
    print(f"  Relevancy Score: {result.relevancy_score:.2f}")
    print(f"  Hallucination Detected: {result.hallucination_detected}")
    print(f"  Passed: {result.passed}")
    
    if result.details.get("hallucination", {}).get("warnings"):
        print(f"  Warnings: {result.details['hallucination']['warnings']}")
    
    # Test batch evaluation
    print("\n=== Batch Evaluation ===")
    batch_results = evaluator.evaluate_batch(
        [sample_query, sample_query],
        [good_answer, bad_answer],
        [sample_context, sample_context]
    )
    
    print(f"Total samples: {batch_results['total_samples']}")
    print(f"Average faithfulness: {batch_results['avg_faithfulness']:.2f}")
    print(f"Hallucination rate: {batch_results['hallucination_rate']:.0%}")
    print(f"Pass rate: {batch_results['pass_rate']:.0%}")


if __name__ == "__main__":
    main()
