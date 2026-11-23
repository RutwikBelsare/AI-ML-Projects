from pydantic import BaseModel

class ErrorDetail(BaseModel):
    code:str
    messgae:str
    request_id:str

class ErrorResponse(BaseModel):
    error:ErrorDetail
    





















