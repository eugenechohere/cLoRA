import asyncio
from models import OpenAIClient


async def main():
    """Example usage of the OpenAI API wrapper."""
    
    # Initialize client (uses OPENAI_API_KEY from environment)
    async with OpenAIClient() as client:
        
        # Example 1: Simple text completion
        print("=" * 50)
        print("Text Completion Example")
        print("=" * 50)
        
        response = await client.simple_text_completion(
            model="gpt-4o",
            prompt="Explain asyncio in Python in one sentence.",
            max_tokens=100
        )
        print(client.extract_text_from_response(response))
        print()
        
        # Example 2: Chat with image URL
        print("=" * 50)
        print("Image Analysis Example")
        print("=" * 50)
        
        # Using a sample image URL
        response = await client.chat_with_images(
            model="gpt-4o",
            text="What's in this image? Describe it briefly.",
            images=["https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"],
            detail="auto",
            max_tokens=200
        )
        print(client.extract_text_from_response(response))


if __name__ == "__main__":
    asyncio.run(main())
