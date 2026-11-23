#------------------------------ Bad Example-----------------------------------

# class PaymentProcessor:
#     def process_payment(self, payment_type, amount):
#         if payment_type == 'credit card':
#             print(f"Processing credit card payment ₹{amount}")
#         elif payment_type == 'upi':
#             print(f"Processing Upi Payment: ₹{amount}")

# processor = PaymentProcessor() 
# processor.process_payment("credit card", 10000)

# processor.process_payment("upi", 2000)

"""Looks fine initially.
But now business says:
"We need to support PayPal."
You modify the existing class:

class PaymentProcessor:

    def process_payment(self, payment_type, amount):

        if payment_type == "credit_card":
            print(f"Processing credit card payment: ₹{amount}")

        elif payment_type == "upi":
            print(f"Processing UPI payment: ₹{amount}")

        elif payment_type == "paypal":
            print(f"Processing PayPal payment: ₹{amount}")

Then tomorrow:

Apple Pay
Google Pay
Net Banking
Crypto
Stripe
Razorpay

Your class becomes:

if ...
elif ...
elif ...
elif ...
elif ...
elif ...
elif ...


Instead of making PaymentProcessor know about every payment method, define a common interface.

In Python, we can use an abstract base class.
"""



# ---------------Good Method ----------------------------------

from abc import ABC, abstractmethod

class PaymentMethod(ABC):

    @abstractmethod
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentMethod):
    def pay(self, amount):
        print(f"Processing credit card payment: ₹{amount}")

class UpiPayment(PaymentMethod):
    def pay(self, amount):
        print(f"Processing credit card payment: {amount}")

class PayPalPayment(PaymentMethod):
    def pay(self, amount):
        print(f"Processing paypal payment of ₹{amount}")

# Now our processor doesn't need to know the specific payment type.

class PaymentProcessor:
    def process(self, payment_method, amount):
        payment_method.pay(amount)

processor = PaymentProcessor('credit card', 1000)
processor.process(CreditCardPayment(),1000)
processor.process(UpiPayment(),200),
processor.processs(PayPalPayment(),400)


 




