
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from utils.logger import logger


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
):
    """Handle HTTP exceptions consistently."""

    logger.warning(
        "HTTP exception: %s %s - Status: %s - Detail: %s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": str(exc.detail),
            "detail": None,
        },
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Handle request validation errors consistently."""

    logger.warning(
        "Request validation failed: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Request validation failed.",
            "detail": exc.errors(),
        },
    )


async def general_exception_handler(
    request: Request,
    exc: Exception,
):
    """Handle unexpected application exceptions."""

    logger.exception(
        "Unexpected error: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected internal server error occurred.",
            "detail": None,
        },
    )
