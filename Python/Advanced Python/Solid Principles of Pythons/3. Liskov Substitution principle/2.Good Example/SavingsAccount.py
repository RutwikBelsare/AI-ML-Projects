from withdrawableAccount import WithdrawableAccount
class SavingsAccount(WithdrawableAccount):
    def __init__(self, amount):
        super().__init__(amount)

    def deposit(self, amount):
        self.balance += amount
        print(f"Amout deposited, current balance {self.balance}")

    def withdraw(self, amount):
        if amount> self.balance:
            print("Cannot withdraw ammount, not enough balance.")
        else:
            self.balance -= amount
            print(f"Successfully Balance updated, Current balance :{self.balance}")
        


