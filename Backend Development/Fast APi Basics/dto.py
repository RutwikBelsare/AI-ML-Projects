from pydantic import BaseModel

class ProductsDTO(BaseModel):
    id: int
    name: str
    price : int = 0
    count : int = 0




