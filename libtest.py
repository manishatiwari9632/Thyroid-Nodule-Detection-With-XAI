from fastapi import UploadFile
import fastapi
import pydantic

print("FastAPI:", fastapi.__version__)
print("Pydantic:", pydantic.__version__)
print("UploadFile:", UploadFile)
print("UploadFile module:", UploadFile.__module__)