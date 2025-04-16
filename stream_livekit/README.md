From this directory:


Create file for environment variables:
```
touch .env
```

Add the following to your new .env file:

```
FIRST_NAME=<Your First Name>
DEEPGRAM_API_KEY=<Your Deepgram API Key>
OPENAI_API_KEY=<Your OpenAI API Key>
CARTESIA_API_KEY=<Your Cartesia API Key>
LIVEKIT_API_KEY=<your API Key>
LIVEKIT_API_SECRET=<your API Secret>
LIVEKIT_URL=wss://unity-gs4xjl87.livekit.cloud
```

The set up the virtual environment:
```bash
uv venv .stream_livekit
source .stream_livekit/bin/activate
uv pip install -r requirements.txt
```

Download model files:
```python
python main.py download-files
```

Run the demo:
```python
python main.py console
```

See this link for more details:
https://docs.livekit.io/agents/start/voice-ai/