from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from backend.api.routes.support import router as support_router
from backend.core.exceptions import AppExceptions
from backend.core.logging_config import configure_logging

import logging
import uuid



configure_logging()
logger = logging.getLogger(__name__)




app = FastAPI(
    title = "Customer Support App",
    version = '1.0.0',
    description = "Ai-powered customer support API using FasAPI and an LLM model"
)
app.include_router(support_router)



@app.get("/health")
def health_check():
    return{
        "status":"ok",
        "service":"genai-customer-support-api"
    }


@app.middleware("http")
async def request_id_middleware(request:Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    logger.info(
        "Request started | request_id = %s | method = %s | path = %s",
        request_id,
        request.method,
        request.url.path
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled Exception | request_id = %s", request_id)
        raise 
    response.headers["X-Request-ID"] = request_id

    logger.info("Request completed | request_id = %s | status=%s", request_id, response.status_code)

    return response

@app.exception_handler(AppExceptions)
async def app_exception_handler(
    request:Request,
    exc: AppExceptions
):
    request_id = getattr(request.state,"request_id","unknown")

    logger.error("Application error | request_id = %s | code = %s | message = %s", request_id, exc.code, exc.message)

    return JSONResponse(status_code = exc.status_code,content={
        "error" : {
            "code": exc.code,
            "message" : exc.message,
            "request_id":request_id
        }
    },)

@app.get("/test-error")
async def test_error():
    raise AppExceptions(
        code = "Test_ERROR",
        message = "This is a test error",
        status_code = 500
    )






