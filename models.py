import asyncio
import base64
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import aiohttp
import time

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
            model: Model name (e.g., "gpt-5", "gpt-4-vision-preview", "gpt-4-turbo")
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


class ConversationManager:
    """
    Manages multi-turn conversations with support for multiple images and text in each turn.
    """
    
    def __init__(self, client: OpenAIClient, system_prompt: Optional[str] = None):
        """
        Initialize the conversation manager.
        
        Args:
            client: OpenAIClient instance to use for API calls
            system_prompt: Optional system prompt to set the context
        """
        self.client = client
        self.messages: List[Dict[str, Any]] = []
        
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
    
    def add_user_message(
        self,
        text: str,
        images: Optional[List[Union[str, Path, bytes]]] = None,
        detail: str = "auto"
    ):
        """
        Add a user message to the conversation.
        
        Args:
            text: Text content of the message
            images: Optional list of images (URLs, file paths, or bytes)
            detail: Image detail level ("low", "high", or "auto")
        """
        content = [{"type": "text", "text": text}]
        
        if images:
            for image in images:
                content.append(self.client.create_image_content(image, detail=detail))
        
        self.messages.append({
            "role": "user",
            "content": content
        })
    
    def add_assistant_message(self, text: str):
        """
        Add an assistant message to the conversation.
        
        Args:
            text: Text content of the assistant's response
        """
        self.messages.append({
            "role": "assistant",
            "content": text
        })
    
    async def send(
        self,
        model: str,
        text: Optional[str] = None,
        images: Optional[List[Union[str, Path, bytes]]] = None,
        detail: str = "auto",
        auto_add_response: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message and get a response. Automatically manages conversation history.
        
        Args:
            model: Model name (e.g., "gpt-5", "gpt-5")
            text: Text to send in this turn (if provided, adds as user message)
            images: Optional list of images to include with this turn
            detail: Image detail level ("low", "high", or "auto")
            auto_add_response: Whether to automatically add the assistant's response to history
            **kwargs: Additional parameters to pass to chat_completion
            
        Returns:
            API response dictionary
        """
        # Add user message if text is provided
        if text is not None:
            self.add_user_message(text, images=images, detail=detail)
        
        # Send the conversation
        response = await self.client.chat_completion(
            model=model,
            messages=self.messages,
            **kwargs
        )
        
        # print(response)
        # Automatically add the assistant's response to history
        if auto_add_response:
            assistant_text = self.client.extract_text_from_response(response)
            self.add_assistant_message(assistant_text)
        
        return response
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the current conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.messages.copy()

    def pop_earliest_turns(self, k: int = 1):
        """
        Remove and return the first k message turns from the conversation history.
        This is a mutating operation.

        Args:
            k: Number of earliest turns to pop.

        Returns:
            List of removed message dictionaries.
        """
        popped = []
        for _ in range(min(k, len(self.messages))):
            popped.append(self.messages.pop(0))
        return popped
        
    def clear_history(self, keep_system_prompt: bool = True):
        """
        Clear the conversation history.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt (if any)
        """
        if keep_system_prompt and self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []

    def chatbot_turn_count(self) -> int:
        """
        Return the number of chatbot turns, i.e., the number of user-assistant pairs exchanged.
        A turn is counted whenever there is a user message followed (anywhere after) by an assistant message.
        The count is the minimum of user and assistant messages.
        
        Returns:
            Number of chatbot turns (user-assistant pairs).
        """
        user_count = sum(1 for msg in self.messages if msg.get("role") == "user")
        assistant_count = sum(1 for msg in self.messages if msg.get("role") == "assistant")
        return min(user_count, assistant_count)

# Example usage
async def example_usage():
    
    # Initialize client (will use OPENAI_API_KEY from environment)
    async with OpenAIClient() as client:
        
        dir = "output/20251025_000643_c7d173a9"
        chunk_size = 3

        max_conv_chatbot_turns = 4

        conversation = ConversationManager(
            client=client,
        )

        from glob import glob

        # Get all screenshot image paths, sorted
        image_paths = sorted(glob(f"{dir}/screenshot_*.png"))

        # Iterate in chunks
        for i in range(0, len(image_paths), chunk_size):
            start_time = time.time()

            images = image_paths[i:i + chunk_size]

            text = "Provided is a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifically. Please be as descriptive possible about all the actions taken. Just give the description without any other chatty text."

            if i != 0:
                text = "Here is a continuation of the previous sequence of actions, provided as a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifically. Please be descriptive about all the actions taken and changes that amde. In your description, do not repeat information or nuances that have already been mentioned in previous turns. Only describe new actions or changes that have been taken since."

            response = await conversation.send(
                model="gpt-5-nano",
                text=text,
                images=images,
                detail="auto",
                reasoning_effort="minimal"
            )

            conversation.messages[-2]["content"] = [{"type": "text", "text": f"Please provide a description of the new actions taken since the previous turn."}]
 
            print(time.time() - start_time)
            if conversation.chatbot_turn_count() >= max_conv_chatbot_turns:
                popped_turns = conversation.pop_earliest_turns(2)
                # popped_assistant_text = popped_turns[0]["content"]


            # print(len(conversation.messages))

            print(f"{client.extract_text_from_response(response)}")
            print("--------------------------------")

   
if __name__ == "__main__":
    asyncio.run(example_usage())

