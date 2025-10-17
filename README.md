# Promo Generator

## How to run
```bash
git clone "https://github.com/jashan7305/promo-generator.git"
cd backend
python -m venv venv

# For Windows:
venv\Scripts\activate

# For Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload
```
## NOTE

This system focuses on speech-based analysis rather than visual content
making it ideal for trailers, interviews, speeches, or dialogue-heavy videos.

For best results, choose a specific and relevant theme instead
of something too vague.
For example:

For avengers.mp4,
suitable themes include:
“save the world”, “action”, or “ambition”.

Everything runs locally using lightweight open-source models
ensuring speed, privacy, and full offline capability

You can try the included avengers.mp4 trailer in the example_video folder,
or use any video with clear dialogues.