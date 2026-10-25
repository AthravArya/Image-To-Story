import logging
import os
import uuid
from pathlib import Path
from typing import Any
from flask import Flask, jsonify, render_template, request
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest, InternalServerError

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, pipeline
from PIL import Image
from gtts import gTTS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", 
)
logger = logging.getLogger(__name__)

logger.info("Loading caption model and processor...")
model_id = "ydshieh/vit-gpt2-coco-en"
image_model = VisionEncoderDecoderModel.from_pretrained(model_id)
image_processor = ViTImageProcessor.from_pretrained(model_id)

logger.info("Loading text-generation model...")
text_gen_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=200,
    min_new_tokens=80,
    temperature=0.8,
    repetition_penalty=1.5,
    no_repeat_ngram_size=3,
)

llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
prompt = PromptTemplate(
    input_variables=["scenario"],
    template = """
You are a creative storyteller.

Write a meaningful story of at least 5-6 sentences based on:
{scenario}

Rules:
- Do not repeat words or phrases
- Make it natural and engaging
- Do not include instructions

Story:
"""
)

chain = prompt | llm | StrOutputParser()


def generate_caption(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
    output_ids = image_model.generate(pixel_values, max_length=16, num_beams=4)
    caption = image_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

def generate_story(caption: str) -> str:
    prompt_text = f"Write a short story about: {caption}\nStory:"
    result = text_gen_pipeline(prompt_text)[0]["generated_text"]
    story = result[len(prompt_text):].strip()
    return story

def generate_audio(text: str) -> str:
    Path("static/audio").mkdir(parents=True, exist_ok=True)
    filename = os.path.join("static/audio", f"{uuid.uuid4().hex}.mp3")
    gTTS(text).save(filename)
    return filename


app = Flask(__name__)

Path("uploads").mkdir(parents=True, exist_ok=True)
Path("static/audio").mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return render_template("index.html")

@app.get("/result")
def result_page():
    return render_template("result.html")

@app.post("/generate")
def generate():
    file: FileStorage | None = request.files.get("image")
    if not file or file.filename == "":
        raise BadRequest("No image file provided.")
    
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in {"jpg", "jpeg", "png", "webp", "gif"}:
        raise BadRequest("Unsupported file type. Allowed: jpg, jpeg, png, webp, gif")
    
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    image_path = os.path.join("uploads", safe_name)
    file.save(image_path)
    logger.info("Image saved: %s", image_path)
    
    try:
        caption = generate_caption(image_path)
        story = generate_story(caption)
        audio_path = generate_audio(story)
        audio_url = "/" + audio_path.replace("\\", "/")
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise InternalServerError("Generation pipeline failed.") from exc
    
    return jsonify({
        "caption": caption,
        "story": story,
        "audio_url": audio_url,
    })
    
@app.errorhandler(BadRequest)
def bad_request(exc: BadRequest) -> tuple[Any, int]:
    return jsonify({"error": exc.description}), 400

@app.errorhandler(InternalServerError)
def server_error(exc: InternalServerError) -> tuple[Any, int]:
    return jsonify({"error": exc.description}), 500


if __name__ == "__main__":
    app.run(debug=True)