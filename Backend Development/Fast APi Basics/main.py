from fastapi import FastAPI, Request
from mockData import products
from dto import ProductsDTO
app = FastAPI()

@app.get("/")
def home():
    return "Hello rutwik Belsare !"

@app.get("/about")
def about():
    return "About Page !"

@app.get("/contact")
def about():
    return "Contact Page !"


@app.get("/products")
def all_products():
    return products

@app.get("/product/{product_id}")
def get_one_product(product_id:int):
    for oneProduct in products:
        if oneProduct.get("id") == product_id:
            return oneProduct

    return{
        "error":"Product not found"
    }


@app.get("/greet")
def greet(request:Request):
    query_params = dict(request.query_params)
    print(query_params)
    return {
        "greet":f"Hi {query_params.get("name")} !, your age is {query_params.get("age")}"

    }

@app.post("/create_product")
def create_product(product_data:ProductsDTO):
    print(product_data)
    return "Data created successfully."



@app.put("/update_product/{product_id}")
def update_product(product_data: ProductsDTO, product_id:int):
    for index, oneProduct in enumerate(products):
        if oneProduct.get("id") == product_id:
            products[index] = product_data.model_dump()
            return {
                "status" : "Product Updated Sucessfully.",
                "products" : product_data
            }
    return {
        "error": f"Not found data for thhis id." 

    }


@app.delete("/delete_product/{product_id}")
def delete_product(product_id:int):
    for index, oneProduct in products:
        if oneProduct.get("id") == products:
            deleted_product = products.pop(index)
            return {
                "status" : f"Product deleted Successfully", 
                "product" : deleted_product
            }
    return{
        "error" : "Product not found for this id ."
    }














