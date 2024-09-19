import os
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import torch
from torch import autocast

from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64 

load_dotenv()

huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device = "cuda"
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=huggingface_api_key)
pipe.to(device)

@app.get("/")
def generate(prompt: str):
    with autocast():
        image = pipe(
                    prompt,
                    guidance_scale=8.5,
                ).images[0]
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())

    return Response(content=imgstr, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, port=8000)
