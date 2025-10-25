from pydantic import BaseModel
import os
import re
import json
import asyncio
from models import OpenAIClient
from datetime import datetime
import dotenv

class Context(BaseModel):
    time: datetime
    username: str
    content: str

    def __str__(self):
        # format to "Friday October 24th 11:40PM"
        formatted_time = self.time.strftime("%A %B %dth %I:%M%p")
        return f"All of this work was done on {formatted_time} by {self.username}:\n\n{self.content}"

PROMPT_FRAGMENTS = ["""
**Critical: mix zoom levels**

    * Some questions should zoom in (micro-level: “what command ran?”).
    * Some questions should zoom out (macro-level: “why are they building this at all?”).
    * This mix is required.

You must cover:

   * **What happened** (step-by-step actions, in order).
   * **Why it matters** (what is the purpose of those actions / what are they working toward).
   * **Current status** (is it working? prototype? production-ready?).
   * **Ownership / responsibility** (who is supposed to do what next, if visible).
   * **Blockers / unanswered questions**.
   * **Impact / next step** (what could happen next based on the work so far).
   * **Risk / policy concerns** if any (privacy, security, etc.).
   * **Summary ** at the end as one of the questions, framed like “What did <person> accomplish?” (change the phrasing)""".strip(),
   """**You must cover high level questions (e.g., What is <person> doing, What was person doing at <time>?, What did <person> accomplish at <time>?, How did <user> accomplish X?, What is a summary of what user did from <time> to <time>?, etc.)**""".strip(),
   """**You must format the questions to be very specific and targeted. Generate a lot of specific questions about the work that was done. They do not need to be high-level or require any thinking (these are questions for recall). For example:

   * At 11PM on Friday, October 24th, what did Eugene do after opening the spreadsheet?
   * What time did Jonathan send a message to Eugene about an API key?
   * What were the endpoints that Sungjin used to train the model when we was working on the code at 11am on October October 23rd?
   * What was the name of the file that Jonathan was working on after he closed the Google Doc at night on October 24th?
   * ...

   **You MUST generate at least 20 questions.**
""".strip(),
]

PROMPT = """You are given an activity log describing what a single person did (the “user”). Your job is to act like a manager or close coworker who did NOT see those logs but wants to understand what happened.

You must do two things in your answer:

1. Write a set of questions that a manager or coworker would realistically ask about that work.
2. Write the answers to those questions based ONLY on the log, plus reasonable inference.

Style and content requirements (follow all of these):

1. **Perspective**

   * Treat the user in the logs as “the person” or by name if given (e.g. “Eugene”).
   * You are NOT that person. You are summarizing and interpreting their work, like a status reviewer.

2. **Question format**

   * Each question should be a bolded header-style question, like:
     `### 1. What did Eugene do in Spotify around 11:40 PM?`
   * Then answer in normal text under it.
   * Write at least 12 questions unless the log is extremely tiny.

3. **Answer style**

   * Answers must sometimes be very concrete and specific:

     * Mention filenames, timestamps, button clicks, exact strings that appeared, commands that ran, etc.
     * Example: “He ran `uv run python screenshot_testing.py`, which saved multiple screenshots like `screenshot_000X.png` and printed `Saved: screenshot_000X.png` in the terminal.”
   * Other answers must be high-level / interpretive:

     * Explain intent, motivation, impact, status, next steps, or risks.
     * Example: “This suggests he’s building tooling to auto-capture work sessions and generate an audit trail.”
   * Mix these two styles across the questions. Some answers should feel like status reporting to leadership. Some should feel like forensic playback.

4. **Inference rules**

   * You ARE allowed to infer reasonable intent from context (e.g. “He highlighted this in red, which likely means it’s unresolved or needs attention.”).
   * You are NOT allowed to make up facts that conflict with the log.
   * If you infer something, state it plainly as inference with language like “This suggests…,” “This implies…,” “Most likely…,” “We don’t see evidence that…”.
   * If the log doesn’t contain the answer, say that clearly (e.g. “We don’t see who else was collaborating on that doc.”).

5. **Scope of questions**
{prompt_fragments}

6. **Voice**

   * Keep it confident and readable, not robotic.
   * Use plain language. You can say things like “This looks like…” or “Classic pattern here is…”.
   * Do NOT use corporate buzzword soup. Avoid purple prose.
   * It should read like a thoughtful engineering manager doing a debrief at midnight.

7. **Timestamps and names**

   * Always anchor major actions to explicit timestamps or time ranges if they’re in the log (e.g. “around 11:40 PM on Oct 24”).
   * Mention file names, playlist names, section headers, etc. exactly as they appeared in the log.
   * If a section in a doc is labeled with someone’s name (like “Jonathan”), call that out and discuss what it implies.

8. **Don’t do**

   * Don’t list the raw events with bullets and stop. You MUST turn them into Q&A with analysis.
   * Don’t speak in first person like “I did X” or “I think Y.”
   * Don’t invent people or tools that aren’t in the log.
   * Don’t assume success unless the log confirms it (for example, only say something “worked” if you saw success indicators like confirmations, saved output, etc.).

9. **Final deliverable structure**

   * Output should be a sequence of sections like:

     ```
     ### 1. [Question here?]
     [Answer paragraph(s).]

     ### 2. [Question here?]
     [Answer...]
     ```
   * No intro, no outro. Just start with question 1.

After these instructions, you will receive a log of activity. Use ONLY that log to generate your Q&A.

If a detail is not in the log, you MUST say you don’t see it in the log."""

extract_prompt = """Here's a bunch of question and answer pairs. Extract the question and answer pairs and put them into a JSON list where the objects contain keys "question" and "answer":"""

def parse_questions(response: str) -> list:
    """
    Parses a string response of the form:

        ### 1. [Question here?]
        [Answer paragraph(s).]

        ### 2. [Question here?]
        [Answer...]

    Strips leading whitespace, removes trailing spaces on lines, normalizes spacing,
    and returns a list of dicts {"question": ..., "answer": ...}

    Sometimes it also looks like (there are thinking tags, and sometimes a separator line):
    <think>
    [thinking...]
    </think>

    ### 1. [Question here?]
    [Answer paragraph(s).]

    ---

    ### 2. [Question here?]
    [Answer...]
    """
    # Remove thinking tags and their contents
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove indentation and trailing spaces from all lines
    # Also filter out lines that are just dashes (separator lines)
    response = "\n".join(
        line.strip() 
        for line in response.strip().splitlines()
        if not re.match(r'^-+$', line.strip())
    )

    # Find all question-answer pairs
    sections = re.split(r"^###\s*\d+\.\s*", response, flags=re.MULTILINE)
    qa_list = []
    for section in sections:
        if not section.strip():
            continue
        # The first line is the question (until a newline), rest is answer
        lines = section.strip().split('\n', 1)
        question = lines[0].strip()
        answer = lines[1].strip() if len(lines) > 1 else ''
        # Remove extra whitespace and spaces at end of each line in answer
        answer = "\n".join(l.rstrip() for l in answer.splitlines()).strip()
        qa_list.append({"question": question, "answer": answer})
    return qa_list

async def generate_synth_data(contexts: list[Context], model: str, prompt_style: str):
    async with OpenAIClient() as client:
        import time

        start_time = time.time()
        contexts_str = "\n".join(str(context) for context in contexts)
        additional_kwargs = {}
        if "openai" in model:
            additional_kwargs = {
                "reasoning_effort": "medium"
            }
        if "qwen" in model:
            additional_kwargs = {
                "reasoning_effort": "default"
            }

        response = await client.simple_text_completion(
            model=model,
            system_prompt=PROMPT.format(prompt_fragments=prompt_style),
            prompt=contexts_str,
            max_completion_tokens=8192,
            temperature=1,
            top_p=0.99,
            **additional_kwargs
        )

        end_time = time.time()
        print(f"Time taken: {(end_time - start_time):.2f} seconds for model {model}")
        return parse_questions(client.extract_text_from_response(response))

async def general_all_prompts(contexts: list[Context], models: list[str], prompt_styles: list[str]):
    requests = []
    for model in models:
        for prompt_style in prompt_styles:
            synth_data = generate_synth_data(contexts, model, prompt_style)
            requests.append(synth_data)
    
    result = await asyncio.gather(*requests)
    return result

if __name__ == "__main__":
    dotenv.load_dotenv()

    context = """A YouTube tab is initially open in a web browser window showing a paused or playing video featuring a person speaking into a RØDE microphone while gesturing with their hands. Subtitles are visible at the bottom of the video with the text, “similar to maybe what evolution has done. That’s why I call pre-training.” The YouTube interface is clearly displayed with the top bar showing the search field and a visible red play icon. The browser is identified as Arc, indicated by the label in the dock at the bottom of the screen, where several application icons such as Finder, Notion, Slack, ChatGPT, and Spotify are also visible. A title at the bottom left corner of the video identifies the speaker and the video title.

Following this, the scene changes to the Spotify desktop app, which fills the screen. A playlist titled “Playlist - eugene” with 64 songs and a total length of 3 hours and 29 minutes is selected from the left sidebar, under the user’s library. The playlist displays its list of songs, each showing the title, artist, album, and the time added, mostly listed as “2 weeks ago.” The active track display at the bottom left corner of this first Spotify frame does not specify a song playing yet.  

The next change shows that playback has begun within Spotify. At the bottom playback bar, the song “SPEED UP” by BENJAMIN RICH is currently playing, as indicated by the highlighted green progress bar moving between timestamps 1:15 and 2:15. The rest of the interface remains the same: the playlist titles and cover remain visible on the left, and the right sidebar continues to show the friend activity feed, listing friends and their recently played tracks. The desktop dock at the bottom confirms that the same Arc browser, Notion, and Spotify applications are active.  

Overall, the sequence depicts a user moving from watching a YouTube video in Arc to interacting with Spotify, selecting and starting playback of a track from a playlist.
--------------------------------
The user continues navigating through the Spotify desktop app without stopping playback of the currently active track, which remains “SPEED UP” by BENJAMIN RICH. The progress bar continues to move slightly from 1:17 to 1:22 as transitions occur, indicating the song keeps playing during these interactions.  

The first visible change is that the user has switched from viewing a playlist to interacting with the account options in the upper-right corner of the interface. A dropdown menu appears below their profile icon, presenting options labeled “Account,” “Profile,” “Support,” “Private session,” “Settings,” “Update Spotify now” (highlighted with a dot to the left), and “Log out.” This indicates the user has clicked their profile avatar to reveal account management actions. The main content area still shows the playlist titled “Playlist - eugene,” now expanded to 65 songs lasting 3 hours and 31 minutes, labeled as “Public Playlist.” The playlist’s header has a solid green background, and the “Added to” notification hovers near the bottom center, likely confirming that a song was successfully saved or added to a collection.  

The user then navigates to their public Spotify profile page, as shown by the profile name “eugene” prominently displayed against a dark blue gradient background. The header includes details showing “7 Public Playlists,” “14 Followers,” and “15 Following.” Below that, personal listening analytics are visible, with a “Top artists this month” section featuring large circular artist portraits, including AJ Vitanza, Tate McRae, keshi, Drake, The Kid LAROI, and Justin Bieber. Below, the “Top tracks this month” section lists songs such as “I Found Out,” “Go Big or Go Home,” “Homemade Dynamite (Feat. Khalid, Post Malone & SZA) – REMIX,” and “INNOCENT,” including their durations (ranging between 2:15 and 3:34). A small label stating “Only visible to you” under the analytics indicates this data is private to the user’s account.  

Next, the user transitions from the profile page to the artist page for “keshi.” The artist’s header banner replaces the previous layout, showing an image of the musician lying on the grass beside a large rock, accompanying the name “keshi” and the “Verified Artist” badge. Displayed metrics indicate 8,257,994 monthly listeners. Several controls appear below, including a large green “Play” button, a “Following” indicator (signifying the user is subscribed to the artist’s updates), and a context menu. The section titled “Popular” presents a track list: “Soft Spot,” “UNDERSTAND,” “WANTCHU,” “LIMBO,” and “beside you,” each accompanied by play counts in the hundreds of millions and track lengths ranging from 2:30 to 3:46. At the bottom, the display includes “Liked Songs” showing “You’ve liked 26 songs” by this artist and an “Artist pick” labeled “WANTCHU,” marked as “Posted by keshi.”  

Throughout all these transitions, the right-hand friend activity panel remains constant, showing contacts, their most recently played songs, and timestamps (“2d” or “3d” ago). The system clock at the top right only advances slightly—from 3:51 PM to about 3:52 PM—during these navigation actions, indicating these events occurred within roughly a one-minute window of use.
--------------------------------
The user remains on the artist page for **keshi** within Spotify at first, continuing to view the artist’s profile and top tracks. The track playback at the bottom of the window, “SPEED UP” by BENJAMIN RICH, continues to play and progresses slightly from 1:24 to 1:30, showing the song is still ongoing. During this time, the main action taken is the removal of an item from the user’s library — specifically, the “Artist pick” for “WANTCHU,” which briefly displays a rectangular pop-up message stating **“Removed from Your Library.”** This notification appears above the “Artist pick” section at the bottom right of the main content area, confirming that the user unliked or removed the song that had been previously saved. Once this action is complete, that small pop-up disappears, and the rest of the interface remains visually consistent.  

Following this, the user shifts away from the Spotify window completely, switching back to the **Arc browser**. The browser is now brought to the foreground, once again displaying the **YouTube** tab containing a paused or actively playing video. The same person as before is visible mid-sentence, speaking into a **RØDE microphone** while gesturing during a podcast discussion. The subtitle at the bottom reads, *“similar to maybe what evolution has done. That’s why I call pre-training.”* The YouTube player remains open in its normal viewing mode, with the upper bar showing the familiar search field and **YouTube logo**, and the bottom Mac dock is visible, confirming the return to the Arc browser window.  

This transition marks a clear workflow shift: the user stops making any further adjustments in Spotify, leaves the music playing in the background, and refocuses their attention on watching the YouTube video within Arc, re-engaging with the content they had viewed earlier.
--------------------------------
The user continues watching the same YouTube video within the Arc browser but interacts with the video timeline to skip forward to a later segment of the conversation. Initially, the footage shows the same speaker as before mid-discussion, but the playback position at the bottom timeline clearly advances from around **12:51** to **1:47:23**, signaled by a visible jump in the red progress bar and the label under the timeline indicating the new topic, “Why self-driving took so long.” Playback resumes for this segment, confirmed by the play icon changing to indicate the video is playing, and subtitles now display a new portion of dialogue beginning with “are not dissimilar to self-driving. What people will often say is that.”  

The next notable change involves a camera switch within the video itself. The viewpoint transitions from the earlier speaker—who remains seated before the same patterned wall and RØDE microphone—to the conversation partner, who is now visible on screen in a mid-shot. This second speaker sits in front of a brick wall lined with bookshelves and another suspended RØDE microphone setup. The subtitles continue in sync with this exchange, now reading “self-driving took so long because the cost of failure is so high.” The overlayed paused-play icon reappears centered on the screen, signaling that the user has either manually paused playback or the video has momentarily been stopped.  

Throughout these actions, no other parts of the interface change—the YouTube layout remains consistent, with the title **“Andrej Karpathy — ‘We’re summoning ghosts, not building animals’”** displayed below, alongside the channel name “Dwarkesh Patel” and view controls such as “Like,” “Share,” “Ask,” and “Download.” The playback bar shows the segment progression marked by pink chapter separators, hinting that the user intentionally navigated between specific sections. The system clock at the top right remains at **3:52 PM**, confirming that these navigations occurred in rapid sequence within the same viewing session.
--------------------------------
The user continues actively interacting with the YouTube interface while the same video remains open and visible in the Arc browser. The video continues showing the podcast guest mid-discussion, with the caption still reading, “self-driving took so long because the cost of failure is so high.”  

The first new action is that the user subscribes to the channel hosting the video. This is evidenced by a visual change to the **“Subscribe”** button located under the channel name—marked by a pink button labeled “Subscribe”—which becomes temporarily animated upon being clicked. A confirmation message appears in the lower-left corner of the screen that reads **“Subscription added”**, confirming that the action was successful. The button and accompanying bell icon next to it reflect the newly subscribed status, showing that this interaction has linked the user’s account to receive updates from the channel.  

After subscribing, the user scrolls downward through the video’s description panel to reveal additional details about the episode. This expands the area below the video title, exposing extended text that had previously been collapsed under the “Show more” section. The full description is now visible and segmented into clear categories, including “EPISODE LINKS,” “SPONSORS,” and “TIMESTAMPS.” Under these sections, multiple hyperlinks appear—for instance, web addresses to a Substack post, Apple Podcasts, Spotify, and sponsor websites such as **labelbox.com**, **mercury.com**, and **gemini.google**. The timestamps correspond to discussion topics from the video, such as “00:00 – AGI is still a decade away,” “00:39 – LLM cognitive deficits,” “01:13 – AGI will blend into 2% GDP growth,” and others, showing that the full list of chapter time markers has loaded as a result of scrolling.  

During this scroll, the lower portion of the right sidebar—showing suggested videos like “Shopify Distinguished Eng (L10)...” and “Please Don’t Download ChatGPT’s Atlas...” —remains visible and static, confirming that the user’s focus is confined to expanding the description area rather than navigating away. The video player continues paused on the same frame, the system clock continues reading **3:52 PM**, and the notification bubble for “Subscription added” is still visible briefly near the bottom left, fading out as the user finishes expanding the episode description.  

These actions show a clear shift from passive viewing to direct engagement—subscribing to the channel and then accessing the detailed textual and linked resources provided alongside the video content.
--------------------------------
After expanding and reading through the episode description, the user scrolls down the YouTube page to move beyond the description area into the **comments section** of the same video. The black background and video player remain fixed at the top while the content below shifts, showing that the user intentionally dragged downward, likely with either the trackpad or mouse scroll, to explore viewer discussions associated with the episode.  

The **first visible change** is that the previously expanded description collapses, leaving a shortened header summarizing the episode once again. Beneath that, the visible section now lists **“1,476 Comments,”** positioned above the input field prompting the viewer to “Add a comment…” The comments populate immediately, indicating that the user has scrolled past the video’s metadata and into the public discussion thread.  

Several comments appear in succession, revealing that the user scrolled continuously down the feed to read reactions from other viewers. The highest-rated comments come from users praising the video’s insights and the guest’s articulation. One comment, posted roughly *eight days ago* by a user named “@integundechatz7872,” jokes about the speed of Karpathy’s speech, saying, **“Karpathy’s tokens/sec is breaking my context window”**, which has **2.1K likes** and **33 replies.** Following immediately below, another user under “@unsupervised-learning” comments about neural encoding, noting a technical insight related to evolution and weight growth—with **100 likes** and **2 replies.**  

The user scrolls slightly further down, loading more comments while the total number of likes and reply counts help contextualize engagement hierarchy. Comments near the top include humorous or admiring remarks such as **“Just realised its not running on 2x lol”** at **2.6K likes** and others expressing appreciation for the clarity and intelligence of the discussion, including **“Almost every sentence this man utters is perfectly put”** (385 likes) and **“Am a simple man. I see Karpathy and Dwarkesh I clear my schedule.”** Each of these entries lists multiple threaded replies visible through expandable counts like “80 replies” or “11 replies,” but none are expanded — the view remains in the main comment feed, suggesting passive reading rather than interaction.  

As the scroll continues, several new user comments come into view beyond the initial few, indicating that the user has progressed farther downward through the feed. Additional reactions appear, such as one highlighting the combination of knowledge and humility (“Karpathy has the great combination of being knowledgeable and working in the field...”) and another referencing a memorable quote about “building ghosts,” acknowledging its precision within the interview context. Further down, more appreciative and reflective comments appear — for example, “What a lovely man. He says everything with a smile” and “The density of thought is astonishing.” Each comment retains its small associated like count and occasional reply link, but the user does not appear to click or expand any of them.  

On the right-hand side, through the entire sequence of scrolling, the **suggested video column** remains fully visible and static, with new thumbnails for related or trending content aligning vertically — titles such as *“AWS outage And ANOTHER AI BROWSER???”*, *“Reid Hoffman on AI, Consciousness, and the Future,”* and *“OpenAI is burning cash.”* None appear selected or hovered over, indicating no navigation outside the comment section.  

In summary, all recent actions involve a deliberate exploration of the viewer comment thread beneath the video, moving from the description section downward to read numerous new comments without interacting with them directly or engaging with any suggested videos. The user’s focus remains entirely within the video’s community section, scrolling to take in written audience reactions and insights about the topic while the video itself remains paused at the top of the page.
--------------------------------
After previously browsing YouTube, the user transitions into the **Slack desktop application**. The workspace visible is labeled **“Cal Hacks 12.0”**, evident from the text at the top of the window. The user lands first on the **Direct Messages** section, identified by the speech bubble icon highlighted along the left sidebar. The column on the left lists three active chats: one with **Jonathan Li**, a personal notes thread labeled **“Eugene Cho (you)”**, and another with **Kandra Chau [Cal Hacks]**. Above these, a search box invites the user to “Find a DM,” paired with a button that reads **“Browse All People.”** The main area at this point is empty except for Slack’s default placeholder illustration — a large, purple 3D message bubble in the center of the screen — indicating no conversation is currently selected.  

Next, the user clicks on **Jonathan Li**’s name in the direct message list, causing the main panel to populate with the full thread between them. This chat history includes both text and media shared earlier. The most recent conversation from “Yesterday” shows several messages from both participants. The earlier part of the thread includes a **Google Docs link** posted by the user at 1:03 AM, followed by a message from Jonathan saying “hello,” then a shared file labeled **“image.png”** from the user shortly after. Below that, casual exchanges appear, including “yo what up,” “the end,” and “brah.” The interface updates dynamically as the selected conversation loads, revealing the name **Jonathan Li** at the top of the chat panel with options for “Messages,” “Add canvas,” and “Files,” as well as a **profile card button** reading “View Profile.” The blurred image preview from the earlier exchange reappears in this step, acting as part of the chat’s shared content.  

After reviewing the conversation, the user sends a **new message**, evident by the addition of new text beneath the previous thread. The message typing area at the bottom becomes active as the user composes and submits the text **“are you done your metaprompt”** at **3:53 PM**, as shown by the timestamp beside the message. This message appears directly below the previous day’s conversation, now grouped under the heading **“Today”** to signify a new session. A green check mark icon appears momentarily beside the message input bar, confirming Slack successfully sent the message.  

In conjunction with this action, the image preview from before is now displayed in higher resolution and clarity — a portrait of a person outdoors in soft light — indicating the image fully loaded during this interaction. The rest of the interface stays constant: the sidebar retains the same message list, the dark purple accent theme of Slack persists, and the Mac menu bar across the top continues to show system icons, Wi-Fi status, and the current time updated slightly to **3:53 PM**.  

Overall, the actions taken consist of the user leaving the previously open browser environment, opening Slack, navigating into a specific direct message thread, reviewing earlier exchanged text and file content, and sending a short new message asking about Jonathan Li’s progress on a “metaprompt.” No other chat threads or interface sections (like channels, files, or activity tabs) are interacted with, indicating a focused one-on-one communication sequence.
--------------------------------
Following the most recent interaction in the Slack workspace, a small but clear update takes place in the direct message thread between the user and **Jonathan Li**. After the user’s earlier message — “are you done your metaprompt” — a **reply arrives from Jonathan Li**, consisting of a brief response: **“no.”** It appears just below the earlier message in the same thread, timestamped at **3:53 PM**, the same time visible in the Mac system clock at the top right of the screen. This confirms that the reply was sent and received almost immediately after the question, showing the two are conversing in real time.  

This new message is accompanied by subtle interface activity indicators. A small reaction count with a single emoji (a green checkmark inside a box) appears below the fresh message, representing either a short acknowledgment or test emoji reaction. The input field at the bottom of the window remains open and ready for further text entry, showing Slack’s default placeholder, **“Message Jonathan Li.”** The formatting bar directly above it still displays all options for styling, file uploads, and emoji reactions, suggesting that no additional text has been typed yet since receiving the reply.  

Aside from the new incoming message, the rest of the Slack interface remains consistent with before. The **direct messages list** continues to show **Jonathan Li** as the active conversation, now accompanied by a snippet preview of the most recent reply, which reads “no,” along with the gray timestamp marker labeled **“Just now.”** The purple sidebar and top bar still indicate that the user is in the **Cal Hacks 12.0** workspace.  

Overall, the single key action here is the **receipt of Jonathan Li’s reply**, confirming an exchange in progress. The new element of live feedback adds dialogue continuity — moving from the user’s question to the partner’s concise answer — with no other interface sections or navigation elements altered since the previous activity.""".split("\n--------------------------------\n")

    contexts = [
        Context(time=datetime(2025, 10, 24, 23, 25+i), username="Eugene", content=c)
        for i, c in enumerate(context)
    ]

    models = [
        "openai/gpt-oss-120b",
        "moonshotai/kimi-k2-instruct-0905",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
    ]
    result = asyncio.run(general_all_prompts(contexts, models, PROMPT_FRAGMENTS))
    json.dump(result, open("synth_data.json", "w"))