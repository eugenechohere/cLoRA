
from models import OpenAIClient, ConversationManager
import asyncio

# Example usage
async def example_usage():
    
    # Initialize client (will use OPENAI_API_KEY from environment)

    async with OpenAIClient() as client:
        
        dir = "output/20251025_155146_f7eee4a6"
        chunk_size = 3

        max_conv_chatbot_turns = 6

        conversation = ConversationManager(
            client=client,
        )

        from glob import glob

        # Get all screenshot image paths, sorted
        image_paths = sorted(glob(f"{dir}/screenshot_*.png"))

        # Iterate in chunks
        for i in range(0, len(image_paths), chunk_size):
            images = image_paths[i:i + chunk_size]

            text = "Provided is a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifically. Please be as descriptive as possible about all the actions taken and changes that are made so that all the necessary context can be included in the description without sounding overbearing. Please be descriptive and explicit about the specific nuances in the context including any relevant text present. Just give the description without any other chatty text."
            if i != 0:
                text = "Here is a continuation of the previous sequence of actions, provided as a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifically. Please be as descriptive as possible about all the actions taken and changes that are made so that all the necessary context can be included in the description without sounding overbearing. Please be descriptive and explicit about the specific nuances in the context, including any relevant text present. In your description, do not repeat information or nuances that have already been mentioned in previous turns. Only describe new actions or changes that have been taken since."

            response = await conversation.send(
                model="gpt-5-chat-latest",
                text=text,
                images=images,
                detail="auto",
                # reasoning_effort="minimal"
            )

            conversation.messages[-2]["content"] = [{"type": "text", "text": f"Please provide a description of the new actions taken since the previous turn."}]
 
            if conversation.chatbot_turn_count() >= max_conv_chatbot_turns:
                conversation.pop_earliest_turns(2)


            print(f"{client.extract_text_from_response(response)}")
            print("--------------------------------")

   
if __name__ == "__main__":
    asyncio.run(example_usage())
