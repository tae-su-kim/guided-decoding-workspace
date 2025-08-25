import typer
import json
from datasets import load_dataset
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

async def recreate_conversation(data, model_name, sem, url):
    async with sem:
        conv = SimpleConversation()
        try:
            conv.add_message({"role": "user", "content": data["prompt"]})
            await run(conv, model_name, url)
        except Exception as e:
            print(e)
            pass
        return conv.messages

@app.command()
def main(
    *,
    dataset_name: Annotated[str, typer.Option("--dataset-name")]="HuggingFaceH4/ultrachat_200k",
    split: Annotated[str, typer.Option("--split")] = "train_sft",
    output_filename: Annotated[str, typer.Option("--output-filename")],
    model_name: Annotated[str, typer.Option("--model-name")] = "Qwen/Qwen3-32B",
    url: Annotated[str, typer.Option("--url")] = "http://localhost:30000/v1/chat/completions",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 384
):
    sem = asyncio.Semaphore(concurrency)
    async def _main():
        dataset = load_dataset(dataset_name, split=split)
        # dataset = dataset.select(range(300))

        futures = []
        for data in dataset:
            future = recreate_conversation(data, model_name, sem, url)
            futures.append(future)

        # Open file for streaming write
        async with aiofiles.open(output_filename, "w") as f:
            await f.write("[\n")  # Start JSON array
            
            first_item = True
            for future in tqdm.tqdm(asyncio.as_completed(futures), total=len(futures)):
                result = await future
                if result:  # Only write if result is not None/empty
                    if not first_item:
                        await f.write(",\n")
                    
                    # Convert single conversation to JSON and write
                    json_line = json.dumps(result, indent=2)
                    # Indent the entire conversation to fit within the array
                    indented_json = "\n".join("  " + line for line in json_line.split("\n"))
                    await f.write(indented_json)
                    first_item = False
            
            await f.write("\n]")  # End JSON array

    asyncio.run(_main())


if __name__ == "__main__":
    app()
