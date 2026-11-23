from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, users, tasks, websocket
from app.middleware.request_id import RequestIDMiddleware

app =FastAPI(
    title = "APIs",
    description = "Secure Api Management Using Rest API",
    version = "1.0.0"
)
# Creating Middlewares:
app.add_middleware(CORSMiddleware, allow_origins = ["*"], allow_credentials = True, allow_methods = ["*"], allow_headers = ["*"],)

app.add_middleware(RequestIDMiddleware)

# Routers
app.include_router(
    auth.router, 
    prefix = "/api/v1"
)

app.include_router(
    users.router,
    prefix = "/api/v1"
)

app.include_router(
    tasks.router,
    prefix = "/api/v1"
)

app.include_router(
    websocket.router,
    prefix = "/api/v1"
)

@app.get("/health")
async def health_check():
    return{
        "Status" : "ok",
        "Service": "Secure Api Management Using Rest API",
        "Version" : "1.0.0"
    }










