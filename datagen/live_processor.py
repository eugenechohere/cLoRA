import asyncio
from datetime import datetime
import requests
from typing import Callable, Optional, List
import os
import dotenv
import threading
import json

from screenshot_testing import ScreenshotCapture
from models import OpenAIClient, ConversationManager
from generate_synth_data import Context, general_all_prompts, PROMPT_FRAGMENTS

dotenv.load_dotenv()


EXAMPLE = """The setting remains within the Slack desktop application, still focused on the **Direct Message conversation** between *Eugene Cho (you)* and *Jonathan Li* under the *Cal Hacks 12.0* workspace. The workspace name and dark purple theme persist, with the app interface retaining its structure — the workspace header, the left-side navigation column for DMs and primary tools, and the conversation panel occupying the majority of the visible area. However, a number of distinct new developments and visual state updates occur since before.

The first noticeable contextual change occurs in the **chat thread area**, reflecting a freshly received reply from Jonathan Li. Below Eugene’s most recent message (“are you done your metaprompt”), a new message authored by *Jonathan Li* has appeared, timestamped **3:53 PM**, directly aligning with the time displayed in the macOS menu bar at the top right of the screen. His message consists solely of the short lowercase word **“no”**. The typical message formatting persists — Jonathan’s username appears in bold, followed by the timestamp in a softer gray, with the sent text immediately underneath in light gray font against Slack’s charcoal background. His response appears in sequence with proper vertical spacing, visually distinguishing this as part of the same conversation thread.

Just below that timestamped response, Slack automatically attaches **emoji reaction counts** to both recent messages. Eugene’s prior message ("are you done your metaprompt") now features **a small emoji reaction icon** — a green square containing a white checkmark overlaid on the same horizontal line as the message bubble. The reaction counter “1” appears directly next to it in a tiny gray font, indicating that exactly one user reacted (most likely Jonathan acknowledging receipt or signaling completion/non-completion status). Additionally, Jonathan’s new “no” message carries that same green checkmark reaction stacked below it on the bottom-right edge of the message bubble, recorded identically in size. The UI representation of this minor interaction is precise and unambiguous, consistent with Slack’s native style for message reactions — subtle, aligned at message mid-height, and semi-illuminated upon hover.

In the **left-hand sidebar**, which previously displayed “You: are you done your metaprompt” under Jonathan Li’s name, that preview text now updates dynamically to **“no”**, signaling that Jonathan’s reply is the most current message in the thread. To the right of his name, the timestamp in faint gray—**“Just now”**—confirms recency. A small green circular marker beside his profile image remains visible, continuing to indicate that he is currently active within the workspace. The sidebar otherwise stays structurally unaltered: beneath Jonathan Li, the entries for “Eugene Cho (you)” and “Kandra Chau [Cal Hacks]” remain as before, but all their indicators remain dimmed and unchanged.

Lower down in the active DM pane, where the **message input field** resides, the state of the text composer is now completely empty—no characters typed, no placeholder text being overwritten, and no active cursor blinking—indicating that Eugene has not started composing a new message since receiving the reply. The text bar retains its functional icons: a gray **“Aa”** button for text formatting sits at the far left, followed by reaction icons arranged sequentially — a smiley face for emoji, an @ symbol for mentions, a paperclip for attachments, a lightning bolt shortcut symbol, and a more-options ellipsis. The **send arrow** in the bottom right corner remains faint gray, implying inactivity until new input is entered.  

Toward the center of the DM area, visually above these new interactions, Slack maintains Eugene’s earlier uploaded image of the man outdoors (the portrait of a curly-haired person in a blue sweater with eyes closed under sunlight). However, this content is unchanged—now serving as a persistent historic artifact in the conversation scroll, which hasn’t visibly moved. The scrollbar on the right-hand edge is positioned high, revealing that the conversation length is short enough to keep all entries fully visible without requiring extra scrolling.

The top navigation bar above the message area remains static, reaffirming the channel context with the name “Jonathan Li” accompanied by his circular profile thumbnail. To the right of his name, smaller utility buttons are still present: “Add canvas,” “Files,” and a plus (+) symbol for further options. All of these remain unclicked. The **workspace global search bar** at the top (“Search Cal Hacks 12.0”) continues to reflect an idle state — no text entry, no recent term pulled down. In the upper-right corner, secondary channel controls such as the “Huddle” dropdown and a trio of icons (for details, call, and window management) remain unaltered, indicating no initiations of calls, canvases, or detailed member views during this timeframe.

Across macOS’s menu bar along the top of the screen, the time has moved forward slightly to **3:53 PM** (matching the message timestamp). Other status icons like Wi‑Fi, Control Center, volume, and battery persist without change, indicating uninterrupted system focus on Slack and no cross-application activity. The desktop background remains completely obscured, meaning the Slack window retains full focus and occupies the screen’s entire viewport without other overlapping windows.

Summarizing the flow of new actions and environmental updates, these include:  
- A **reply** from Jonathan Li appearing in contrast to the previously one-sided conversation, consisting of the short message “no,” timestamped precisely to current system time.  
- **Reaction indicators** becoming visible on both Eugene Cho’s last message and Jonathan’s newly received response, showing a green checkmark emoji with a count of “1.”  
- The **left sidebar conversation preview** updating dynamically to reflect Jonathan’s new message content.  
- The **input field** shifting back to an idle state, confirming no ongoing composition from Eugene.  
- All UI components such as hover tooltips, system bars, tab menus, timestamps, and general Slack layout remaining constant aside from those message-level content changes.  

Overall, the new activity centers entirely around the addition of Jonathan Li’s concise response and its corresponding reaction annotations, signaling the continuation of an active real-time chat exchange without any external workspace navigation or UI reconfiguration."""

FIRST_TURN_PROMPT = f"Provided is a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifically. Please be as descriptive as possible about all the actions taken and changes that are made so that all the necessary context can be included in the description without sounding overbearing. Please be descriptive and explicit about the specific nuances in the context including any relevant text present and UI elements that are present and deemed relevant to jot down, including actions and elements that don't explicitly impact the webpage, e.g. side UI elements,scrolling, text text being highlighted by the user, tool tips, button presses/button state changes, etc.. Please be as explicit and descriptive and verbose as possible about the current context. Just give the description without any other chatty text.\n\nProvided is an example of how descriptive you should be writing wise: {EXAMPLE}"
TURN_PROMPT = f"Here is a continuation of the previous sequence of actions, provided as a sequence of frames of a screen. Describe all the actions that are taken throughout the frames, without mentioning frames specifically. Please be as descriptive as possible about all the actions taken and changes that are made so that all the necessary context can be included in the description without sounding overbearing. Please be descriptive and explicit about the specific nuances in the context, including any relevant text present UI elements that are present and deemed relevant to jot down, including actions and elements that don't explicitly impact the webpage, e.g. side UI elements, scrolling, text being highlighted by the user, tool tips, button presses/button state changes, etc... Please be as explicit and descriptive and verbose as possible about the current context. In your description, do not repeat information or nuances that have already been mentioned in previous turns. Only describe new actions or changes that have been taken since. \n\nProvided is an example of how descriptive you should be writing wise: {EXAMPLE}"


class LiveDataProcessor(ScreenshotCapture):
    """
    Extends ScreenshotCapture to process screenshots live and generate synthetic data.
    
    Pipeline:
    1. Every 5 screenshots → create a Context chunk
    2. When 4 Context chunks accumulated → run generate_synth_data
    3. After that, every new Context chunk → drop oldest, keep sliding window of 4, run generate_synth_data
    4. Collect all generated data
    5. After 5 batches of data generations → call callback
    
    Uses a queue-based system to ensure no data is lost.
    """
    
    def __init__(
        self,
        username: str,
        vlm_model: str = "gpt-5-chat-latest",
        qa_models: Optional[List[str]] = None,
        prompt_fragments: Optional[List[str]] = None,
        repeats: int = 5,
        callback: Optional[Callable[[List[dict]], None]] = None,
        screenshots_per_chunk: int = 5,
        context_window_size: int = 4,
        batches_before_callback: int = 5,
    ):
        """
        Initialize the live data processor.
        
        Args:
            username: Username for Context metadata
            vlm_model: Vision-language model for context generation
            qa_models: List of models for Q&A generation
            prompt_fragments: Prompt fragments for synthetic data generation
            repeats: Number of repeats for general_all_prompts
            callback: Function to call after batches_before_callback batches (receives list of all generated data)
            screenshots_per_chunk: Number of screenshots before creating a Context chunk (default: 5)
            context_window_size: Number of Context chunks to keep in sliding window (default: 4)
            batches_before_callback: Number of data generation batches before calling callback (default: 5)
        """
        super().__init__()
        
        self.username = username
        self.vlm_model = vlm_model
        self.qa_models = qa_models or [
            "openai/gpt-oss-120b",
            "moonshotai/kimi-k2-instruct-0905",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32b",
        ]
        self.prompt_fragments = prompt_fragments or PROMPT_FRAGMENTS
        self.repeats = repeats
        self.callback = callback
        self.screenshots_per_chunk = screenshots_per_chunk
        self.context_window_size = context_window_size
        self.batches_before_callback = batches_before_callback
        
        # Buffers and state
        self.screenshot_buffer: List[str] = []  # Paths to screenshots
        self.contexts: List[Context] = []  # Context chunks (sliding window)
        self.all_generated_data: List[dict] = []  # All generated Q&A pairs
        self.generation_batch_count = 0  # Number of times we've run generate_synth_data
        
        # Queue for processing batches of screenshots (no data loss)
        self.screenshot_queue: Optional[asyncio.Queue] = None
        
        # Queue for synthetic data generation (runs independently)
        self.synth_data_queue: Optional[asyncio.Queue] = None
        
        # For context generation
        self.client: Optional[OpenAIClient] = None
        self.conversation: Optional[ConversationManager] = None
        
        # Event loop and worker tasks
        self.loop = None
        self.screenshot_worker_task = None
        self.synth_data_worker_task = None
        self.stop_event = threading.Event()
        
    async def initialize_async(self):
        """Initialize async components."""
        self.client = OpenAIClient()
        await self.client.__aenter__()
        self.conversation = ConversationManager(client=self.client)
        
        # Create the queue for screenshot batches
        self.screenshot_queue = asyncio.Queue()
        
        # Create the queue for synthetic data generation
        self.synth_data_queue = asyncio.Queue()
        
        # Start the screenshot processing worker task
        self.screenshot_worker_task = asyncio.create_task(self.process_screenshot_queue_worker())
        print("[LiveProcessor] Screenshot worker task started")
        
        # Start the synthetic data generation worker task
        self.synth_data_worker_task = asyncio.create_task(self.process_synth_data_queue_worker())
        print("[LiveProcessor] Synthetic data worker task started")
        
    async def cleanup_async(self):
        """Clean up async components."""
        # Stop the screenshot worker task
        if self.screenshot_worker_task:
            self.screenshot_worker_task.cancel()
            try:
                await self.screenshot_worker_task
            except asyncio.CancelledError:
                pass
        
        # Stop the synthetic data worker task
        if self.synth_data_worker_task:
            self.synth_data_worker_task.cancel()
            try:
                await self.synth_data_worker_task
            except asyncio.CancelledError:
                pass
        
        # Clean up client
        if self.client:
            await self.client.__aexit__(None, None, None)
    
    def take_screenshot(self):
        """Override to add screenshot to buffer and trigger processing."""
        # Call parent's take_screenshot
        super().take_screenshot()
        
        # Get the path of the screenshot we just took
        screenshot_path = str(self.output_dir / f"screenshot_{self.count - 1:04d}.png")
        self.screenshot_buffer.append(screenshot_path)
        
        print(f"[LiveProcessor] Screenshot {self.count} captured. Buffer size: {len(self.screenshot_buffer)}")
        
        # Check if we have enough screenshots to enqueue a batch
        if len(self.screenshot_buffer) >= self.screenshots_per_chunk:
            # Extract a batch and put it on the queue
            batch = self.screenshot_buffer[:self.screenshots_per_chunk]
            self.screenshot_buffer = self.screenshot_buffer[self.screenshots_per_chunk:]
            
            print(f"[LiveProcessor] Enqueuing batch of {len(batch)} screenshots (queue size: {self.screenshot_queue.qsize() if self.screenshot_queue else 0})")
            
            # Put the batch on the queue (non-blocking from the screenshot thread)
            if self.screenshot_queue is not None:
                asyncio.run_coroutine_threadsafe(
                    self.screenshot_queue.put(batch),
                    self.loop
                )
    
    async def process_screenshot_queue_worker(self):
        """Worker task that continuously processes screenshot batches from the queue."""
        print("[LiveProcessor] Screenshot queue worker started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for a batch from the queue (with timeout to allow checking stop_event)
                try:
                    batch = await asyncio.wait_for(self.screenshot_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                print(f"[LiveProcessor] Processing batch of {len(batch)} screenshots from queue...")
                
                # Process this batch
                await self.process_screenshot_batch(batch)
                
                # Mark task as done
                self.screenshot_queue.task_done()
                
            except asyncio.CancelledError:
                print("[LiveProcessor] Screenshot queue worker cancelled")
                break
            except Exception as e:
                print(f"[LiveProcessor] Error in screenshot queue worker: {e}")
                import traceback
                traceback.print_exc()
        
        print("[LiveProcessor] Screenshot queue worker stopped")
    
    async def process_synth_data_queue_worker(self):
        """Worker task that continuously processes synthetic data generation from the queue."""
        print("[LiveProcessor] Synth data queue worker started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for a context batch from the queue (with timeout to allow checking stop_event)
                try:
                    contexts_to_process = await asyncio.wait_for(self.synth_data_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                print(f"[LiveProcessor] Generating synthetic data from {len(contexts_to_process)} contexts (from synth queue)...")
                
                # Generate synthetic data from this batch of contexts
                await self.generate_synthetic_data(contexts_to_process)
                
                # Mark task as done
                self.synth_data_queue.task_done()
                
            except asyncio.CancelledError:
                print("[LiveProcessor] Synth data queue worker cancelled")
                break
            except Exception as e:
                print(f"[LiveProcessor] Error in synth data queue worker: {e}")
                import traceback
                traceback.print_exc()
        
        print("[LiveProcessor] Synth data queue worker stopped")
    
    async def process_screenshot_batch(self, screenshots_to_process: List[str]):
        """Process a batch of screenshots to create a Context chunk."""
        
        try:
            print(f"\n[LiveProcessor] Processing {len(screenshots_to_process)} screenshots to create Context chunk...")
            
            # Determine if this is the first turn
            is_first_turn = len(self.contexts) == 0
            text = FIRST_TURN_PROMPT if is_first_turn else TURN_PROMPT
            
            # Generate context description using VLM
            response = await self.conversation.send(
                model=self.vlm_model,
                text=text,
                images=screenshots_to_process,
                detail="auto",
            )
            
            # Simplify the message history
            self.conversation.messages[-2]["content"] = [
                {"type": "text", "text": "Please provide a description of the new actions taken since the previous turn."}
            ]
            
            # Pop earliest turns if conversation is getting too long
            max_conv_chatbot_turns = 6
            if self.conversation.chatbot_turn_count() >= max_conv_chatbot_turns:
                self.conversation.pop_earliest_turns(2)
            
            # Extract text from response
            description = self.client.extract_text_from_response(response)
            
            # Calculate average timestamp from screenshots
            image_timestamps = [os.path.getmtime(img_path) for img_path in screenshots_to_process]
            avg_timestamp = sum(image_timestamps) / len(image_timestamps)
            avg_datetime = datetime.fromtimestamp(avg_timestamp)
            
            # Create Context object
            context = Context(time=avg_datetime, username=self.username, content=description)
            
            print(f"[LiveProcessor] Created Context chunk at {avg_datetime}")
            print(f"[LiveProcessor] Context preview: {description[:200]}...")
            
            # Add to contexts (sliding window)
            self.contexts.append(context)
            
            # Check if we should enqueue synthetic data generation
            should_generate = False
            
            if len(self.contexts) == self.context_window_size:
                # First time we have enough contexts
                print(f"[LiveProcessor] Reached {self.context_window_size} contexts. Enqueuing for synthetic data generation...")
                should_generate = True
            elif len(self.contexts) > self.context_window_size:
                # Sliding window: drop the oldest context
                self.contexts = self.contexts[-self.context_window_size:]
                print(f"[LiveProcessor] Sliding window: keeping last {self.context_window_size} contexts. Enqueuing for synthetic data generation...")
                should_generate = True
            else:
                print(f"[LiveProcessor] Have {len(self.contexts)}/{self.context_window_size} contexts. Waiting for more...")
            
            # Enqueue synthetic data generation if we have enough contexts (non-blocking)
            if should_generate:
                # Make a copy of the current contexts to pass to the synth data queue
                contexts_copy = self.contexts.copy()
                await self.synth_data_queue.put(contexts_copy)
                print(f"[LiveProcessor] Enqueued {len(contexts_copy)} contexts for synthetic data generation (synth queue size: {self.synth_data_queue.qsize()})")
        
        except Exception as e:
            print(f"[LiveProcessor] Error processing screenshot batch: {e}")
            import traceback
            traceback.print_exc()
    
    async def generate_synthetic_data(self, contexts: List[Context]):
        """Generate synthetic Q&A data from given context window."""
        print(f"\n[LiveProcessor] Generating synthetic data from {len(contexts)} contexts...")
        
        try:
            # Generate Q&A pairs
            result = await general_all_prompts(
                contexts,
                self.qa_models,
                self.prompt_fragments,
                repeats=self.repeats
            )
            
            print(f"[LiveProcessor] Generated {len(result)} Q&A pairs")
            
            # Add to all generated data
            self.all_generated_data.extend(result)
            self.generation_batch_count += 1
            
            print(f"[LiveProcessor] Total data points: {len(self.all_generated_data)}")
            print(f"[LiveProcessor] Generation batches: {self.generation_batch_count}/{self.batches_before_callback}")
            
            # Check if we should call the callback
            if self.generation_batch_count >= self.batches_before_callback and self.callback:
                print(f"\n[LiveProcessor] Reached {self.batches_before_callback} batches. Calling callback...")
                
                # Call callback with all generated data
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(self.all_generated_data.copy())
                else:
                    self.callback(self.all_generated_data.copy())
                
                # Reset for next round
                self.all_generated_data = []
                self.generation_batch_count = 0
                print("[LiveProcessor] Callback completed. Reset for next round.")
        
        except Exception as e:
            print(f"[LiveProcessor] Error generating synthetic data: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_event_loop(self):
        """Run the event loop in a background thread."""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
    
    def run(self):
        """Run the live processor with async support."""
        print(f"\n{'='*80}")
        print("LIVE DATA PROCESSOR STARTED")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Username: {self.username}")
        print(f"Screenshots per chunk: {self.screenshots_per_chunk}")
        print(f"Context window size: {self.context_window_size}")
        print(f"Batches before callback: {self.batches_before_callback}")
        print(f"{'='*80}\n")
        
        # Create event loop in a separate thread
        self.loop = asyncio.new_event_loop()
        
        # Start event loop in background thread
        loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        loop_thread.start()
        
        # Wait a moment for loop to start
        import time
        time.sleep(0.1)
        
        # Initialize async components
        future = asyncio.run_coroutine_threadsafe(self.initialize_async(), self.loop)
        future.result()  # Wait for initialization to complete
        
        print("[LiveProcessor] Async initialization complete. Ready to capture screenshots.\n")
        
        try:
            # Start parent's run (this will block in main thread)
            super().run()
        except KeyboardInterrupt:
            print("\n\n[LiveProcessor] Stopping...")
        finally:
            # Signal stop
            self.stop_event.set()
            
            # Clean up async components
            print("[LiveProcessor] Cleaning up...")
            if self.loop and self.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self.cleanup_async(), self.loop)
                try:
                    future.result(timeout=5.0)
                except Exception as e:
                    print(f"[LiveProcessor] Cleanup error: {e}")
                
                # Stop the loop
                self.loop.call_soon_threadsafe(self.loop.stop)
            
            # Wait for loop thread to finish
            loop_thread.join(timeout=2.0)
            
            print(f"\n{'='*80}")
            print("LIVE DATA PROCESSOR STOPPED")
            print(f"{'='*80}")
            print(f"Total contexts created: {len(self.contexts)}")
            print(f"Total data points generated: {len(self.all_generated_data)}")
            print(f"Generation batches completed: {self.generation_batch_count}")
            if self.screenshot_queue:
                print(f"Screenshot batches remaining in queue: {self.screenshot_queue.qsize()}")
            if self.synth_data_queue:
                print(f"Synth data batches remaining in queue: {self.synth_data_queue.qsize()}")
            print(f"{'='*80}\n")


index = 0
async def example_callback(data: List[dict]):
    """Example callback function that could call an endpoint."""
    global index
    print(f"\n{'='*80}")
    print("CALLBACK TRIGGERED")
    print(f"{'='*80}")
    print(f"Received {len(data)} Q&A pairs")
    print("\nFirst 3 examples:")
    for i, item in enumerate(data[:3], 1):
        print(f"\n{i}. Q: {item.get('question', 'N/A')}")
        print(f"   A: {item.get('answer', 'N/A')[:100]}...")
    
    # TODO: Call your endpoint here
    # Example:
    # async with aiohttp.ClientSession() as session:
    #     async with session.post('https://your-endpoint.com/api/data', json=data) as resp:
    #         print(f"Endpoint responded with: {resp.status}")

    json.dump(data, open(f"output/result_{index}.json", "w"))
    index += 1

    # @app.post("/upload")
    # async def upload_example(payloads: List[Dict[str, str]] = Body(...)) -> Dict[str, Any]:
    # https://8d01de42c0ff.ngrok-free.app/

    data = [{"prompt": item["question"], "completion": item["answer"]} for item in data]
    result = requests.post("https://8d01de42c0ff.ngrok-free.app/upload", json=data, headers={
        "Content-Type": "application/json", "ngrok-skip-browser-warning": "0"
    })

    print(result.json())
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Example usage
    processor = LiveDataProcessor(
        username="Eugene",
        # vlm_model="gpt-5-chat-latest",
        vlm_model="meta-llama/llama-4-maverick-17b-128e-instruct",
        qa_models=[
            "openai/gpt-oss-120b",
            "moonshotai/kimi-k2-instruct-0905",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "qwen/qwen3-32b",
        ],
        prompt_fragments=PROMPT_FRAGMENTS,
        repeats=1,
        callback=example_callback,
        screenshots_per_chunk=5,

        # number of context chunks before generating synthetic data
        context_window_size=2,
        
        batches_before_callback=1,
    )
    
    processor.run()

