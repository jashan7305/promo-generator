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

## Note
This system focuses on speech-based analysis rather than visual content<br>
making it ideal for trailers, interviews, speeches, or dialogue-heavy videos.<br>

For best results, choose a specific and relevant theme instead<br>
of something too vague.<br> 
For example:<br>

For avengers.mp4,<br>
suitable themes include:<br>
“save the world”, “action”, or “ambition”.<br>

Everything runs locally using lightweight open-source models,<br>
ensuring speed, privacy, and full offline capability.<br>

You can try the included avengers.mp4 trailer in the example_video folder,<br>
or use any video with clear dialogues.<br>