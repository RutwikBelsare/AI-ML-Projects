products = [
    {"id" : 1,"name":"Mobile","phone":"7769074787", "price": 700, "ratings" : 4.2},
    {"id" : 2,"name":"Laptop","phone":"7769074723", "price":450, "ratings":3.5}
]

sorted_by_price = sorted(products, key =lambda p:p["price"])
sorted_by_ratings = sorted(products, key = lambda p:p['ratings'], reverse = True)
print()
print(sorted_by_price)
print()
print(sorted_by_ratings)