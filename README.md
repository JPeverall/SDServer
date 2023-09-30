# Stable Diffusion Server README

This README provides an overview and instructions on how to use the provided stable diffusion server as a REST API. It is a FastAPI-based server that utilizes the `CompVis/stable-diffusion-v1-4` model (other models could be used) running on a private PC. The server uses the model to generate images based on the user's input recieved via a GET request.

In this configuration, it will accept a prompt via a GET request on a public URL, or from the localhost. The [NGROK](https://ngrok.com/docs/secure-tunnels/tunnels/http-tunnels/) free http tunneling service can be used to forward the request from a public address to the local port, but you can use any tunneling service. 

## Prerequisites

You will need the following packages installed:

- FastAPI
- torch
- torchvision
- nltk
- Pydantic
- Uvicorn (or similar for running the server in the local environment)
- CUDA (recommended if using a supported GPU to speed up processing)

## Files

There are two main files:

1. server.py: This file contains the main FastAPI server application that hosts the stable diffusion model.
2. extraction.py: This file contains a script to extract nouns from a given prompt using NLTK.

## Usage

1. Run `server.py` to start the FastAPI server. By default, it will be hosted on `localhost:8000`. You may need to configure your `auth_token` in a separate `auth_token.py` file, which should contain a single variable named `auth_token` with your API key.
   
   ```
   python server.py
         or
   uvicorn server:app
   ```

2. Access the server by sending a GET request to the root endpoint with a `prompt` query parameter. The server will return a generated image based on the given prompt. Example:

   ```
   http://localhost:8000?prompt=forest
   ```
In the current configuation, a `permanent_promt` is combined with the prompt received in the GET request to provide more consistent style.

3. If desired, the `extraction.py` file can be used as a standalone script to extract nouns from a given text prompt. It is not neccessary for the server to run, but can be useful for processing user input.

In the current configuration, it is used in the `server.py` file to extract words based on their similarity to a list of `reference_words`.

## Configuration

The `server.py` file contains a list of allowed origins for CORS. Update this list as needed to match your desired configuration:

```python
allowed_origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",
    "Add your website URL here",
]
```

## Notes

The generated images are saved as `testimage.png` in the current directory for local validation. The server also returns the generated image as a base64-encoded PNG image, which can be used directly in web applications or other clients that support base64-encoded images.

