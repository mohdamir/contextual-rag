from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import re

class SanitizeJSONMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-type") == "application/json":
            body_bytes = await request.body()
            try:
                body_text = body_bytes.decode("utf-8")
                # Clean control characters
                sanitized_text = re.sub(r'[\x00-\x1f\x7f]', ' ', body_text)
                request._body = sanitized_text.encode("utf-8")
            except Exception as e:
                return JSONResponse(status_code=400, content={"detail": "Invalid JSON"})
        return await call_next(request)