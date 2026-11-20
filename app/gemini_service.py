"""
Google Gemini API integration service using the modern `google-genai` SDK.
Handles model communication, multi-turn chat generation, streaming,
embeddings generation, and API key verification.
"""
import os
import logging
from typing import List, Dict, Any, Generator, Optional
from google import genai
from google.genai import types
from app.config import (
    get_gemini_api_key,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

class GeminiService:
    """Encapsulates interaction with Google Gemini models."""
    
    def __init__(self, api_key: Optional[str] = None):
        self._explicit_key = api_key
        self._client: Optional[genai.Client] = None
    
    @property
    def api_key(self) -> str:
        """Get the current API key (explicitly passed or from environment)."""
        return self._explicit_key or get_gemini_api_key()
    
    def get_client(self) -> genai.Client:
        """Get or initialize the Google GenAI client."""
        key = self.api_key
        if not key:
            raise ValueError(
                "Gemini API key is not configured. Please set the GEMINI_API_KEY environment variable "
                "or enter it in the Settings panel."
            )
        # Re-initialize client if key changes
        if self._client is None or getattr(self._client, "_api_key", None) != key:
            self._client = genai.Client(api_key=key)
            self._client._api_key = key
        return self._client
    
    def is_configured(self) -> bool:
        """Check whether a Gemini API key is available."""
        return bool(self.api_key)
    
    def validate_api_key(self, key_to_test: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate an API key by making a lightweight test call.
        Returns a dict with 'valid': bool, 'message': str.
        """
        key = key_to_test or self.api_key
        if not key:
            return {"valid": False, "message": "No API key provided."}
        
        try:
            temp_client = genai.Client(api_key=key)
            # Try fast test call with gemini-3.5-flash-lite, fallback to gemini-3.8-flash
            try:
                response = temp_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents="Ping. Reply with 'pong'.",
                    config=types.GenerateContentConfig(max_output_tokens=10)
                )
            except Exception:
                response = temp_client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents="Ping. Reply with 'pong'.",
                    config=types.GenerateContentConfig(max_output_tokens=10)
                )
            return {
                "valid": True,
                "message": "Gemini API key verified successfully!",
                "reply": getattr(response, "text", "pong")
            }
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return {
                "valid": False,
                "message": f"Validation failed: {str(e)}"
            }
    
    def generate_chat_response(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate a conversational response from Gemini with multi-turn memory.
        
        Args:
            user_message: The latest prompt from the user.
            conversation_history: List of prior turns [{'role': 'user'|'model', 'parts': '...'}]
            system_instruction: Guiding prompt for persona and task boundaries.
            model: Gemini model identifier (e.g. gemini-3.8-flash).
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens in generated completion.
        """
        client = self.get_client()
        target_model = model or DEFAULT_CHAT_MODEL
        
        # Build contents list preserving multi-turn context
        contents = []
        if conversation_history:
            for turn in conversation_history:
                role = "user" if turn.get("role") in ["user", "human"] else "model"
                text = turn.get("content") or turn.get("text", "")
                if text:
                    contents.append(types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=text)]
                    ))
        
        # Append latest user message
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        ))
        
        # Build config
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction if system_instruction else None
        )
        
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=contents,
                config=config
            )
        except Exception as e:
            # If target model hit 503 or quota spike, try fallback model
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                logger.warning(f"Model {target_model} busy, falling back to gemini-3.5-flash-lite: {e}")
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=contents,
                    config=config
                )
            else:
                raise e
        
        return response.text or ""
    
    def generate_chat_stream(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """
        Stream conversational tokens incrementally for Server-Sent Events (SSE).
        """
        client = self.get_client()
        target_model = model or DEFAULT_CHAT_MODEL
        
        contents = []
        if conversation_history:
            for turn in conversation_history:
                role = "user" if turn.get("role") in ["user", "human"] else "model"
                text = turn.get("content") or turn.get("text", "")
                if text:
                    contents.append(types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=text)]
                    ))
        
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        ))
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction if system_instruction else None
        )
        
        try:
            response_stream = client.models.generate_content_stream(
                model=target_model,
                contents=contents,
                config=config
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                logger.warning(f"Streaming on {target_model} hit 503, retrying with gemini-3.5-flash-lite: {e}")
                fallback_stream = client.models.generate_content_stream(
                    model="gemini-3.5-flash-lite",
                    contents=contents,
                    config=config
                )
                for chunk in fallback_stream:
                    if chunk.text:
                        yield chunk.text
            else:
                raise e
                yield chunk.text

    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate vector embeddings using Gemini's embedding model.
        Batches calls to maintain low latency and adhere to rate limits.
        """
        client = self.get_client()
        embedding_model = model or DEFAULT_EMBEDDING_MODEL
        
        results: List[List[float]] = []
        # Batch size of 16 for optimal API throughput
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Gemini embedding API accepts string or list of strings
                response = client.models.embed_content(
                    model=embedding_model,
                    contents=batch
                )
                
                # Check response.embeddings
                if hasattr(response, "embeddings") and response.embeddings:
                    for emb in response.embeddings:
                        results.append(list(emb.values))
                elif hasattr(response, "embedding") and response.embedding:
                    results.append(list(response.embedding.values))
                else:
                    raise ValueError("No embeddings returned by API.")
            except Exception as e:
                logger.error(f"Error embedding batch {i}-{i+len(batch)}: {e}")
                raise e
        
        return results

# Singleton instance for application-wide reuse
gemini_service = GeminiService()
