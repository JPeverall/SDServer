import os
from dotenv import load_dotenv
import nltk
import nltk.corpus
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import torch
from diffusers import StableDiffusionPipeline
from io import BytesIO
from torch.cuda.amp import autocast
import base64
load_dotenv()
auth_token = os.getenv('auth_token')
print(auth_token)

app = FastAPI()

allowed_origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

device = "cuda"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token = auth_token)
pipe.to(device)

@app.get("/")
def generate(prompt: str):
    permanent_prompt = "(pencil drawing), pen drawing, sketch, Tolkien art style, (book art)"
    negative = "character, person, animal, face"
    full_prompt = permanent_prompt + prompt
    with autocast():
        image = pipe(full_prompt, negative_prompt=negative, guidance_scale=8.5).images[0]

    image.save("testimage.png")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())
    print(full_prompt)

    return Response(content=imgstr, media_type="image/png")