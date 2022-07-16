import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from environment_registry import EnvironmentRegistry
import calls.environment.configure
import calls.environment.reset
import calls.environment.step
import calls.environment.delete
import requests

app = FastAPI()

origins = [
    # "*",
    "http://localhost:1234",
    "http://localhost:8000",
    "http://localhost:*",
    "https://prl-api.herokuapp.com/*",
    "https://prl-frontend.herokuapp.com/*"
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
app.include_router(calls.environment.configure.router)
app.include_router(calls.environment.reset.router)
app.include_router(calls.environment.step.router)
app.include_router(calls.environment.delete.router)

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