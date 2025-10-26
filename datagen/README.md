# CalHacks Continual Learning

A Python project with an OpenAI API wrapper that supports sending image requests to any arbitrary model using asyncio and aiohttp.

## Features

- 🚀 Asynchronous API calls using `asyncio` and `aiohttp`
- 🖼️ Support for multiple image input formats:
  - Image URLs (http/https)
  - Local file paths
  - Raw image bytes
- 🔄 Automatic base64 encoding for local images
- 🎯 Support for any OpenAI model (GPT-4o, GPT-4 Vision, etc.)
- 🛠️ Simple and flexible API
- 📝 Type hints for better IDE support

## Installation

```bash
# Install dependencies
uv sync
```

## Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Basic Text Completion

```python
import asyncio
from models import OpenAIClient

async def main():
    async with OpenAIClient() as client:
        response = await client.simple_text_completion(
            model="gpt-4o",
            prompt="What is asyncio?",
            max_tokens=100
        )
        print(client.extract_text_from_response(response))

asyncio.run(main())
```

### Image Analysis (URL)

```python
async with OpenAIClient() as client:
    response = await client.chat_with_images(
        model="gpt-4o",
        text="What's in this image?",
        images=["https://example.com/image.jpg"],
        detail="auto",
        max_tokens=300
    )
    print(client.extract_text_from_response(response))
```

### Multiple Local Images

```python
async with OpenAIClient() as client:
    response = await client.chat_with_images(
        model="gpt-4o",
        text="Compare these images. What are the differences?",
        images=["path/to/image1.jpg", "path/to/image2.jpg"],
        detail="high",
        max_tokens=500
    )
    print(client.extract_text_from_response(response))
```

### Custom Messages with Images

```python
async with OpenAIClient() as client:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                client.create_image_content("path/to/image.jpg", detail="high")
            ]
        }
    ]
    
    response = await client.chat_completion(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    print(client.extract_text_from_response(response))
```

### Using Raw Image Bytes

```python
async with OpenAIClient() as client:
    with open("image.jpg", "rb") as f:
        image_bytes = f.read()
    
    response = await client.chat_with_images(
        model="gpt-4o",
        text="Describe this image",
        images=[image_bytes],
        max_tokens=200
    )
    print(client.extract_text_from_response(response))
```

## API Reference

### OpenAIClient

Main client class for interacting with the OpenAI API.

#### Constructor

```python
OpenAIClient(api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1")
```

- `api_key`: OpenAI API key (defaults to `OPENAI_API_KEY` env var)
- `base_url`: Base URL for API endpoint

#### Methods

##### `chat_completion(model, messages, **kwargs)`
Make a chat completion request with full control over messages.

##### `chat_with_images(model, text, images, detail="auto", **kwargs)`
Make a chat completion request with images.

- `images`: List of image URLs, file paths, or bytes
- `detail`: "low", "high", or "auto" (controls image resolution)

##### `simple_text_completion(model, prompt, system_prompt=None, **kwargs)`
Make a simple text-only completion request.

##### `create_image_content(image, detail="auto")`
Create an image content object from URL, file path, or bytes.

##### `extract_text_from_response(response)`
Extract text content from API response.

## Supported Models

The wrapper works with any OpenAI model that supports the Chat Completions API:

- `gpt-4o` - Latest multimodal model
- `gpt-4o-mini` - Faster, cheaper version
- `gpt-4-turbo` - Previous generation with vision
- `gpt-4-vision-preview` - Vision-specific model
- `gpt-4` - Standard GPT-4
- `gpt-3.5-turbo` - Faster, cheaper option

## Image Detail Levels

- `low`: 512px x 512px (faster, cheaper, less detail)
- `high`: Detailed image analysis (slower, more expensive)
- `auto`: Let the model decide based on image content

## Running Examples

```bash
# Run the main example
python main.py

# Or use uv
uv run main.py
```

## Development

```bash
# Add new dependencies
uv add <package-name>

# Update dependencies
uv sync
```

## License

MIT

