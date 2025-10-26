from typing import Tuple
from models import OpenAIClient
from glob import glob
from models import OpenAIClient, ConversationManager
from generate_synth_data import Context, general_all_prompts, PROMPT_FRAGMENTS
from datetime import datetime
import os
import json
import asyncio 

class JudgeValidator:
    """
    A validator class that compares string values for equality using both
    string comparison and AI-powered semantic comparison.
    """
    
    def __init__(self, client: OpenAIClient):
        """
        Initialize the JudgeValidator.
        
        Args:
            client: OpenAIClient instance to use for model-based comparison
        """
        self.client = client
    
    async def compare_values(self, x: str, y: str) -> Tuple[bool, str]:
        """
        Compare two string values for equality.
        
        First attempts exact string equality. If that fails, uses GPT-4o to
        determine if the values are semantically equivalent.
        
        Args:
            x: First string to compare
            y: Second string to compare
            
        Returns:
            Tuple of (is_equal: bool, method: str) where method is either
            "string-equality" or "model-judge"
        """
        # Step 1: Try string equality check
        if x == y:
            return (True, "string-equality")
        
        # Step 2: Use GPT-4o to compare the values
        prompt = f"""Compare the following two values and determine if they are semantically equivalent or represent the same meaning, even if they differ in exact wording.

Value X: {x}
Value Y: {y}

Are these two values equivalent in meaning? Respond with only "YES" or "NO"."""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = await self.client.chat_completion(
            model="gpt-5-mini",
            messages=messages,
            reasoning_effort="minimal"
        )
        
        # Extract the response text
        answer = self.client.extract_text_from_response(response).strip().upper()
        
        # Determine if the model considers them equal
        is_equal = "YES" in answer
        
        return (is_equal, "model-judge")


PROMPT = ""


async def run_evaluation():

    contexts = []

    async with OpenAIClient() as client:
        dir = "imgs/eval"
        chunk_size = 3
        max_conv_chatbot_turns = 6

        conversation = ConversationManager(
            client=client,
        )

        # Get all screenshot image paths, sorted
        image_paths = sorted(glob(f"{dir}/screenshot_*.png"))

        # Iterate in chunks
        for i in range(0, len(image_paths), chunk_size):
            images = image_paths[i:i + chunk_size]

            text = "Provided is a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifically. Please be as descriptive as possible about all the actions taken and changes that are made so that all the necessary context can be included in the description without sounding overbearing. Please be descriptive and explicit about the specific nuances in the context including any relevant text present and UI elements that are present and deemed relevant to jot down. Please be as explicit and descriptive and verbose as possible about the current context. Just give the description without any other chatty text."
            if i != 0:
                text = "Here is a continuation of the previous sequence of actions, provided as a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifically. Please be as descriptive as possible about all the actions taken and changes that are made so that all the necessary context can be included in the description without sounding overbearing. Please be descriptive and explicit about the specific nuances in the context, including any relevant text present UI elements that are present and deemed relevant to jot down. Please be as explicit and descriptive and verbose as possible about the current context. In your description, do not repeat information or nuances that have already been mentioned in previous turns. Only describe new actions or changes that have been taken since."

            response = await conversation.send(
                model="gpt-5-chat-latest",
                text=text,
                images=images,
                detail="auto",
            )

            conversation.messages[-2]["content"] = [{"type": "text", "text": f"Please provide a description of the new actions taken since the previous turn."}]
 
            if conversation.chatbot_turn_count() >= max_conv_chatbot_turns:
                conversation.pop_earliest_turns(2)


            print(f"{client.extract_text_from_response(response)}")
            print("--------------------------------")

            # Calculate average datetime from image file timestamps
            image_timestamps = [os.path.getmtime(img_path) for img_path in images]
            avg_timestamp = sum(image_timestamps) / len(image_timestamps)
            avg_datetime = datetime.fromtimestamp(avg_timestamp)
            
            contexts.append(Context(time=avg_datetime, username="Eugene", content=client.extract_text_from_response(response)))



    models = [
        "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
    ]
    result = await general_all_prompts(contexts, models, PROMPT_FRAGMENTS)
    json.dump(result, open("synth_data.json", "w"))


if __name__ == "__main__":

    asyncio.run(run_evaluation())