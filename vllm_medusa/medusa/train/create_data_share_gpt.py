import typer
import json
from typing_extensions import Annotated
import httpx
import tqdm
import asyncio
import aiofiles

app = typer.Typer()


client = httpx.AsyncClient(timeout=None)

# Simple conversation class to replace the deprecated transformers Conversation
class SimpleConversation:
    def __init__(self):
        self.messages = []
    
    def add_message(self, message):
        self.messages.append(message)

async def run(conv: SimpleConversation, model_name: str, url: str):
    payload = {
        "model":model_name,
        "messages": conv.messages,
        "temperature": 0.0
    }
    response = await client.post(url, json=payload)
    content = response.json()
    message = content["choices"][0]["message"]
    message.pop("name", None)
    conv.add_message(message)




def fix_source(source):
    if source and source[0]["from"] == "gpt":
        # Skip if GPT is first to talk
        source = source[1:]
    new_source = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        new_source.append({"role": role, "content": content})
    return new_source


async def recreate_conversation(conversation, model_name, sem, url):
    async with sem:
        conv = SimpleConversation()
        try:
            for message in conversation[::2]:
                assert message["role"] == "user"
                conv.add_message(message)
                await run(conv, model_name, url)
        except Exception as e:
            print(e)
            pass
        return conv.messages

async def write_result_to_file(file_handle, result, is_first):
    """Write a single result to the JSON file asynchronously"""
    if not is_first:
        await file_handle.write(",\n")
    json_str = json.dumps(result, indent=4, ensure_ascii=False)
    # Indent the entire JSON object to align with array formatting
    indented_json = '\n'.join('    ' + line for line in json_str.split('\n'))
    await file_handle.write(indented_json)

@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")],
    output_filename: Annotated[str, typer.Option("--output-filename")],
    model_name: Annotated[str, typer.Option("--model-name")] = "Qwen/Qwen3-32B",
    url: Annotated[str, typer.Option("--url")] = "http://localhost:30000/v1/chat/completions",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 384
):
    sem = asyncio.Semaphore(concurrency)
    
    async def _main():
        with open(input_filename, "r") as f:
            input_data = json.loads(f.read())
        conversations = [fix_source(source["conversations"]) for source in input_data]

        # Open output file for writing
        async with aiofiles.open(output_filename, "w", encoding="utf-8") as f:
            await f.write("[\n")
            
            futures = []
            for conversation in conversations:
                future = recreate_conversation(conversation, model_name, sem, url)
                futures.append(future)

            # Process results as they complete and write to file immediately
            completed_count = 0
            is_first = True
            
            for future in tqdm.tqdm(asyncio.as_completed(futures), total=len(futures), desc="Processing conversations"):
                try:
                    result = await future
                    await write_result_to_file(f, result, is_first)
                    is_first = False
                    completed_count += 1
                except Exception as e:
                    print(f"Error processing conversation: {e}")
                    continue
            
            await f.write("\n]")
            
        print(f"Completed processing {completed_count} conversations")
        
    asyncio.run(_main())


if __name__ == "__main__":
    app()
