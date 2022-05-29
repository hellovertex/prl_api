import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from environment_registry import EnvironmentRegistry
import src.calls.environment.configure
import src.calls.environment.reset
import src.calls.environment.step
import src.calls.environment.delete
import requests

app = FastAPI()

origins = [
    "http://localhost:1234",
    "http://localhost:8000",
    "http://localhost:*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.backend = EnvironmentRegistry()

# register api calls
app.include_router(src.calls.environment.configure.router)
app.include_router(src.calls.environment.reset.router)
app.include_router(src.calls.environment.step.router)
app.include_router(src.calls.environment.delete.router)

@app.get("/")
async def root():
    return {"message": "Hello Poker"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
#
#     ploads = {'n_players': 6,
#               'starting_stack_size': 2000}
#
#     r = requests.post(
#         url='http://localhost:8000/environment/configure/',
#         params=ploads)
#
#     print(r.text)