import os
from dotenv import load_dotenv
import nltk  # Natural Language Toolkit, used for processing human language data
import nltk.corpus
from fastapi import FastAPI, Response  # FastAPI framework, for building APIs
from fastapi.middleware.cors import CORSMiddleware  # Middleware for handling Cross Origin Resource Sharing
import torch  # PyTorch, a deep learning framework
from diffusers import StableDiffusionPipeline  # Pipeline for stable diffusion, likely custom module
from io import BytesIO  # Module to manage binary stream
from torch.cuda.amp import autocast  # Module for automatic mixed precision training
import base64  # Module for working with Base64 encoding
from extraction import splitPrompt, extract_nouns_adjectives, is_descriptive  # Custom extraction module

# Loading environment variables
load_dotenv()
auth_token = os.getenv('auth_token')  # Getting auth token from environment variable

app = FastAPI()  # Creating a FastAPI instance

# List of allowed origins for cross origin resource sharing
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "https://jaesolutionsexpressserver.azurewebsites.net",
]

# Adding CORS middleware to the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

device = "cuda"  # Setting device to CUDA for GPU acceleration
model_id = "CompVis/stable-diffusion-v1-4"  # Specifying the model ID
# Initializing the pipeline from a pretrained model
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)
pipe.to(device)  # Moving the pipeline to the specified device

@app.get("/")
def generate(prompt: str):
    paragraph = splitPrompt(prompt)  # Splitting the prompt using custom function
    nouns_adjectives = extract_nouns_adjectives(paragraph)  # Extracting nouns and adjectives from the paragraph
    nouns_adjectives_str = ', '.join(nouns_adjectives)  # Joining extracted words with commas
    permanent_prompt = "(pencil drawing), work of art, pen drawing, sketch, Tolkien art style, (book art)" # Helps maintain general desired style of returned images.
    negative = "face, bad drawing, sloppy"  # Specifying negative words to avoid in the generated image
    full_prompt = f"{permanent_prompt} {nouns_adjectives_str}"  # Forming the full prompt
    with autocast():  # Enabling automatic mixed precision
        image = pipe(full_prompt, negative_prompt=negative, guidance_scale=8.5).images[0]  # Generating image using the pipeline

    image.save("testimage.png")  # Saving the generated image, for local validation.
    buffer = BytesIO()  # Creating a BytesIO object for storing image in memory
    image.save(buffer, format="PNG")  # Saving image to the buffer in PNG format
    imgstr = base64.b64encode(buffer.getvalue())  # Encoding the buffered image to Base64
    print(full_prompt)  # Printing the full prompt to the console
    
    # Returning the Base64 encoded image as a response with media type as image/png
    return Response(content=imgstr, media_type="image/png")
