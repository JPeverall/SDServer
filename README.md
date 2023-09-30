# Stable Diffusion Server README

This README provides an overview and instructions on how to use the provided stable diffusion server. The server is a FastAPI-based server that utilizes the `CompVis/stable-diffusion-v1-4` model from OpenAI's API. The server uses the model to generate images based on the user's input.

## Prerequisites

You will need the following packages installed:

- FastAPI
- torch
- torchvision
- nltk
- Pydantic

## Files

There are two main files:

1. server.py: This file contains the main FastAPI server application that hosts the stable diffusion model.
2. pltk.py: This file contains a script to extract nouns from a given prompt using NLTK.

## Usage

1. Run `server.py` to start the FastAPI server. By default, it will be hosted on `localhost:8000`. You may need to configure your `auth_token` in a separate `auth_token.py` file, which should contain a single variable named `auth_token` with your API key.
   
   ```
   python server.py
   ```

2. Access the server by sending a GET request to the root endpoint with a `prompt` query parameter. The server will return a generated image based on the given prompt. Example:

   ```
   http://localhost:8000?prompt=forest
   ```

3. The `pltk.py` file can be used as a standalone script to extract nouns from a given text prompt. It is not directly related to the server but can be useful for processing user input. To run it, simply execute the script:

   ```
   python pltk.py
   ```

## Configuration

The `server.py` file contains a list of allowed origins for CORS. Update this list as needed to match your desired configuration:

```python
allowed_origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",
]
```

## Notes

The generated images are saved as `testimage.png` in the current directory. The server also returns the generated image as a base64-encoded PNG image, which can be used directly in web applications or other clients that support base64-encoded images.