"""
server/api_errors.py
====================
SDS error schema.  Every error response has the shape

    {"error": "<machine_code>", "message": "<human text>",
     "field": "<offending field or null>", "reference_id": "<uuid>"}

HTTP codes used: 400 bad_request, 401 unauthorized, 403 forbidden,
404 not_found, 409 conflict, 500 internal_error.
"""

import uuid


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field

    def to_response(self):
        body = {
            "error": self.code,
            "message": self.message,
            "field": self.field,
            "reference_id": str(uuid.uuid4()),
        }
        return body, self.status


def bad_request(message, field=None, code="bad_request"):
    return ApiError(400, code, message, field)


def unauthorized(message="A valid X-Session-Token header is required"):
    return ApiError(401, "unauthorized", message)


def forbidden(message, code="forbidden"):
    return ApiError(403, code, message)


def not_found(message, code="not_found"):
    return ApiError(404, code, message)


def conflict(message, code="conflict"):
    return ApiError(409, code, message)
