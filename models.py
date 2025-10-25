import asyncio
import base64
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import aiohttp


class OpenAIClient:
    """
    A basic OpenAI API wrapper with support for sending image requests to any arbitrary model.
    Uses asyncio and aiohttp for asynchronous operations.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1"):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If not provided, will try to read from OPENAI_API_KEY env var.
            base_url: Base URL for the API endpoint. Defaults to OpenAI's API.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    @staticmethod
    def encode_image_from_path(image_path: Union[str, Path]) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Base64-encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    @staticmethod
    def encode_image_from_bytes(image_bytes: bytes) -> str:
        """
        Encode image bytes to base64.
        
        Args:
            image_bytes: Raw image bytes.
            
        Returns:
            Base64-encoded string of the image.
        """
        return base64.b64encode(image_bytes).decode("utf-8")
    
    def create_image_content(
        self,
        image: Union[str, Path, bytes],
        detail: str = "auto"
    ) -> Dict[str, Any]:
        """
        Create an image content object for the API request.
        
        Args:
            image: Can be:
                - A URL string (http:// or https://)
                - A file path (str or Path)
                - Raw image bytes
            detail: Image detail level ("low", "high", or "auto")
            
        Returns:
            A dictionary representing the image content.
        """
        if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
            # Image URL
            return {
                "type": "image_url",
                "image_url": {
                    "url": image,
                    "detail": detail
                }
            }
        elif isinstance(image, bytes):
            # Raw bytes
            base64_image = self.encode_image_from_bytes(image)
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail
                }
            }
        else:
            # File path
            base64_image = self.encode_image_from_path(image)
            # Try to detect mime type from extension
            ext = Path(image).suffix.lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }.get(ext, "image/jpeg")
            
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}",
                    "detail": detail
                }
            }
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a chat completion request.
        
        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo")
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response dictionary
        """
        await self._ensure_session()
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        url = f"{self.base_url}/chat/completions"
        
        async with self._session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()
    
    async def chat_with_images(
        self,
        model: str,
        text: str,
        images: List[Union[str, Path, bytes]],
        detail: str = "auto",
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a chat completion request with images.
        
        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4-vision-preview")
            text: Text prompt/question about the images
            images: List of images (URLs, file paths, or bytes)
            detail: Image detail level ("low", "high", or "auto")
            system_prompt: Optional system prompt
            **kwargs: Additional parameters to pass to chat_completion
            
        Returns:
            API response dictionary
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Build user message with text and images
        content = [{"type": "text", "text": text}]
        
        for image in images:
            content.append(self.create_image_content(image, detail=detail))
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return await self.chat_completion(model=model, messages=messages, **kwargs)
    
    async def simple_text_completion(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a simple text-only chat completion request.
        
        Args:
            model: Model name
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters to pass to chat_completion
            
        Returns:
            API response dictionary
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return await self.chat_completion(model=model, messages=messages, **kwargs)
    
    @staticmethod
    def extract_text_from_response(response: Dict[str, Any]) -> str:
        """
        Extract the text content from an API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Extracted text content
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to extract text from response: {e}")


# Example usage
async def example_usage():
    
    # Initialize client (will use OPENAI_API_KEY from environment)
    async with OpenAIClient() as client:
        
        dir = "imgs/random_highlights"

        # Example 3: Chat with multiple images from file paths
        # (Uncomment if you have local images)
        print("Example 3: Chat with multiple local images")
        response = await client.chat_with_images(
            model="gpt-5",
            text="Provided is a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifcally. Please be as descriptive as possible, being explicit about each action taken",
            images=[f"{dir}/screenshot_000{i}.png" for i in range(0,9)],
            detail="auto",
            reasoning_effort="low",
        )

        print(client.extract_text_from_response(response))
        print()
        
   
if __name__ == "__main__":
    asyncio.run(example_usage())

