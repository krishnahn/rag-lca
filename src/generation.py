"""
LLM Generation Module for LCA RAG Application.

This module handles:
- LLM integration with OpenRouter API
- Custom prompts for grounded responses
- Multi-source answer generation
- Citation and source tracking
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# LlamaIndex imports
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai_like import OpenAILike

from tqdm import tqdm

# Local imports
from src.retrieval import RetrievedContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# RAG Prompt Templates
RAG_SYSTEM_PROMPT = """You are an expert Life Cycle Assessment (LCA) assistant. Your role is to provide accurate, well-sourced answers about LCA methodologies, environmental impacts, and sustainability analysis.

CRITICAL RULES:
1. ONLY answer based on the provided context. Do not use any external knowledge.
2. If the context doesn't contain enough information to answer, clearly state "I don't have enough information in the provided documents to answer this question."
3. ALWAYS cite your sources using the format [Source: filename].
4. When information comes from multiple sources, synthesize it and cite all relevant sources.
5. For numerical data (like GWP values, emissions factors), quote the exact values from the sources.
6. Preserve the structure of information - if data comes from a table, present it in a structured format.
7. For mind map content, maintain the hierarchical relationships in your explanation.

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide supporting details with citations
- If relevant, include any calculations or data from the sources
- End with a brief summary if the answer is complex
"""

RAG_USER_PROMPT_TEMPLATE = """Based on the following context from LCA documents, please answer the question.

CONTEXT:
{context}

QUESTION: {query}

Please provide a comprehensive answer based ONLY on the context above. Remember to cite sources using [Source: filename] format."""


@dataclass
class LLMConfig:
    """Configuration for LLM."""
    model_name: str = "google/gemini-2.0-flash-001"
    api_key: Optional[str] = None  # Falls back to OPENROUTER_API_KEY env var
    api_base: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: float = 120.0


@dataclass
class GenerationResult:
    """Container for generation results."""
    answer: str
    sources: List[str]
    contexts_used: List[RetrievedContext]
    model: str
    tokens_used: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OpenRouterLLM:
    """
    Wrapper for OpenRouter API integration.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._llm = None

    def _get_llm(self) -> OpenAILike:
        """Lazy load the LLM."""
        if self._llm is None:
            try:
                api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY environment variable not set. Get one at https://openrouter.ai/keys")
                self._llm = OpenAILike(
                    model=self.config.model_name,
                    api_key=api_key,
                    api_base=self.config.api_base,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    is_chat_model=True,
                )
                logger.info(f"Initialized OpenRouter with model: {self.config.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenRouter: {e}")
                raise
        return self._llm

    def get_llm(self) -> OpenAILike:
        """Get the LLM instance."""
        return self._get_llm()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        llm = self._get_llm()

        messages = []
        if system_prompt:
            messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
        messages.append(ChatMessage(role=MessageRole.USER, content=prompt))

        response = llm.chat(messages)
        return response.message.content

    def check_availability(self) -> bool:
        """Check if OpenRouter API is available."""
        try:
            llm = self._get_llm()
            response = llm.complete("Say 'OK' if you're working.")
            return bool(response.text)
        except Exception as e:
            logger.warning(f"OpenRouter not available: {e}")
            return False


class PromptBuilder:
    """
    Builds prompts for RAG generation.
    """
    
    def __init__(
        self,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        user_prompt_template: str = RAG_USER_PROMPT_TEMPLATE
    ):
        """
        Initialize the prompt builder.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt_template: Template for user prompts
        """
        self.system_prompt = system_prompt
        self.user_prompt_template = PromptTemplate(user_prompt_template)
        
    def build_prompt(
        self,
        query: str,
        context: str,
        contexts: Optional[List[RetrievedContext]] = None
    ) -> Tuple[str, str]:
        """
        Build the full prompt for generation.
        
        Args:
            query: User query
            context: Formatted context string
            contexts: Optional list of RetrievedContext objects
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Format the user prompt
        user_prompt = self.user_prompt_template.format(
            context=context,
            query=query
        )
        
        return self.system_prompt, user_prompt
    
    def build_multi_source_prompt(
        self,
        query: str,
        contexts: List[RetrievedContext]
    ) -> Tuple[str, str]:
        """
        Build a prompt that emphasizes multi-source synthesis.
        
        Args:
            query: User query
            contexts: List of RetrievedContext objects
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Group contexts by source
        by_source = {}
        for ctx in contexts:
            source = ctx.source.split("/")[-1].split("\\")[-1]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(ctx)
        
        # Format context with clear source attribution
        context_parts = []
        for source, ctxs in by_source.items():
            context_parts.append(f"\n## From {source}:")
            for ctx in ctxs:
                content_label = f"[{ctx.content_type.upper()}]"
                context_parts.append(f"{content_label}\n{ctx.text}")
        
        formatted_context = "\n".join(context_parts)
        
        # Enhanced prompt for multi-source
        multi_source_note = f"\n\nNote: Information is available from {len(by_source)} different sources. Please synthesize information from all relevant sources in your answer."
        
        user_prompt = self.user_prompt_template.format(
            context=formatted_context + multi_source_note,
            query=query
        )
        
        return self.system_prompt, user_prompt


class RAGGenerator:
    """
    Complete RAG generation pipeline.
    
    Handles context-grounded response generation.
    """
    
    def __init__(
        self,
        llm: OpenRouterLLM,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        """
        Initialize the RAG generator.
        
        Args:
            llm: OpenRouterLLM instance
            prompt_builder: Optional PromptBuilder instance
        """
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        
    def generate(
        self,
        query: str,
        context: str,
        contexts: List[RetrievedContext]
    ) -> GenerationResult:
        """
        Generate a response for a query.
        
        Args:
            query: User query
            context: Formatted context string
            contexts: List of RetrievedContext objects
            
        Returns:
            GenerationResult object
        """
        # Build prompts
        if len(set(ctx.source for ctx in contexts)) > 1:
            # Multi-source query
            system_prompt, user_prompt = self.prompt_builder.build_multi_source_prompt(
                query, contexts
            )
        else:
            system_prompt, user_prompt = self.prompt_builder.build_prompt(
                query, context, contexts
            )
        
        # Generate response
        try:
            answer = self.llm.generate(user_prompt, system_prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            answer = f"Error generating response: {e}"
        
        # Extract unique sources
        sources = list(set(ctx.source for ctx in contexts))
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            contexts_used=contexts,
            model=self.llm.config.model_name,
            metadata={
                "query": query,
                "num_contexts": len(contexts)
            }
        )
    
    def generate_with_followup(
        self,
        query: str,
        context: str,
        contexts: List[RetrievedContext],
        conversation_history: Optional[List[Tuple[str, str]]] = None
    ) -> GenerationResult:
        """
        Generate with conversation history for follow-up questions.
        
        Args:
            query: Current query
            context: Formatted context
            contexts: Retrieved contexts
            conversation_history: List of (query, answer) tuples
            
        Returns:
            GenerationResult object
        """
        # Include conversation history in prompt
        history_text = ""
        if conversation_history:
            history_parts = []
            for prev_query, prev_answer in conversation_history[-3:]:  # Last 3 exchanges
                history_parts.append(f"Previous Q: {prev_query}")
                history_parts.append(f"Previous A: {prev_answer[:200]}...")
            history_text = "\n\nConversation History:\n" + "\n".join(history_parts)
        
        # Build prompt with history
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            query + history_text, 
            context, 
            contexts
        )
        
        # Generate
        answer = self.llm.generate(user_prompt, system_prompt)
        
        sources = list(set(ctx.source for ctx in contexts))
        
        return GenerationResult(
            answer=answer,
            sources=sources,
            contexts_used=contexts,
            model=self.llm.config.model_name,
            metadata={
                "query": query,
                "num_contexts": len(contexts),
                "has_history": bool(conversation_history)
            }
        )


def main():
    """Test the generation module."""
    # Check Ollama availability
    config = LLMConfig(
        model_name="google/gemini-2.0-flash-001",
        temperature=0.1
    )
    
    llm = OpenRouterLLM(config)
    
    print("\n=== Checking OpenRouter Availability ===")
    if not llm.check_availability():
        print("OpenRouter is not available. Please ensure:")
        print("1. OPENROUTER_API_KEY is set (https://openrouter.ai/keys)")
        print("\nRunning with mock responses for testing...")
        
        # Create mock response for testing
        class MockLLM:
            def __init__(self):
                self.config = config
            
            def generate(self, prompt, system_prompt=None):
                return """Based on the provided context, here's what I found:

Steel production has a Global Warming Potential (GWP) of approximately 2.1 kg CO2 equivalent per kilogram [Source: materials_impact.pdf].

The impact data table shows:
- Steel: 2.1 kg CO2 eq, 25 MJ energy
- Aluminum: 8.1 kg CO2 eq, 155 MJ energy [Source: impact_data.xlsx]

This means steel has a significantly lower carbon footprint compared to aluminum, making it a more environmentally favorable choice from a GWP perspective.

Summary: Steel's environmental impact (2.1 kg CO2 eq/kg) is about 4 times lower than aluminum (8.1 kg CO2 eq/kg)."""
        
        llm = MockLLM()
    else:
        print("OpenRouter is available!")
    
    # Create sample contexts
    sample_contexts = [
        RetrievedContext(
            text="Steel production has a Global Warming Potential of 2.0-2.5 kg CO2 equivalent per kg.",
            score=0.92,
            source="materials_impact.pdf",
            content_type="text",
            metadata={}
        ),
        RetrievedContext(
            text="| Material | GWP (kg CO2 eq) | Energy (MJ) |\n|----------|----------------|-------------|\n| Steel | 2.1 | 25 |\n| Aluminum | 8.1 | 155 |",
            score=0.88,
            source="impact_data.xlsx",
            content_type="table",
            metadata={}
        )
    ]
    
    # Format context
    context_parts = []
    for ctx in sample_contexts:
        context_parts.append(f"[From {ctx.source}]")
        context_parts.append(f"Type: {ctx.content_type}")
        context_parts.append(ctx.text)
        context_parts.append("")
    
    formatted_context = "\n".join(context_parts)
    
    # Create generator
    generator = RAGGenerator(llm)
    
    # Test generation
    print("\n=== Testing Generation ===")
    query = "What is the environmental impact of steel production?"
    print(f"Query: {query}")
    
    result = generator.generate(query, formatted_context, sample_contexts)
    
    print(f"\n--- Generated Answer ---")
    print(result.answer)
    
    print(f"\n--- Sources Used ---")
    for source in result.sources:
        print(f"  - {source}")
    
    print(f"\n--- Metadata ---")
    print(f"  Model: {result.model}")
    print(f"  Contexts used: {result.metadata.get('num_contexts')}")


if __name__ == "__main__":
    main()
