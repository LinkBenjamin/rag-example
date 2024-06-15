# Llama3 Social Media Manager

I post lots of stuff to youtube, but the part that takes time is all the tedious social media stuff... writing a good description, adding the right hashtags, picking out short-form clips that I could pull from the longer video, crafting a clickbait-y title to grab attention... 

🤔 What if I could get AI to do all that FOR me???

## How to try this out

First, install [Ollama](https://www.ollama.com/) on your box and make sure it's running.

Then go to your command line and ask it to 

`ollama pull llama3`

This might take a bit, it's about 4 GB to download.

Next, you'll need to set up the python program:

```bash
python -m venv .venv

source .venv/bin/activate # substitute this with .venv/Scripts/activate in Windows!

pip install -r requirements.txt
```

and then you need to make a couple edits in `main.py` (which I've helpfully grouped just below the imports for easy changing!):

```python
MODEL_ID = 'llama3'                     # No need to change this unless you have a different model running
BASE_URL = 'http://127.0.0.1:11434'     # See above
VIDEO_ID = '***********'                # This is the Unique string ID at the end of a Youtube Video's URL.

def invocations(retrieval_chain):

    # Invocations is the final step in the python program running.  Add in as many of these as you like, edit them to do
    # whatever you want.

    response1 = retrieval_chain.invoke({"input": "Create a summary of this message that's less than 800 characters long.  Then add several hashtags that would be appropriate if this were the youtube description of the video, in order to maximize its social media reach."})

    print(response1['answer'])

    response2 = retrieval_chain.invoke({"input": "Create a clickbait style title for the message based on its overall theme."})

    print(response2['answer'])

    response3 = retrieval_chain.invoke({"input": "Locate at least 3 potential quotable snippets within the message that could make good short-form video content.  Provide ONLY the snippets, do not explain why you selected them."})

    print(response3['answer'])
```

After you've edited these parts to your desire, just run 

`python main.py`

and watch the magic happen!