from user import User
from item import Item


class Order:
    def __init__(self, user, item, item_quant):
        self.user = user
        self.item = item
        self.item_quant = item_quant
    
    def cost(self):
        value = self.item_quant*self.item.item_price
        return value

if __name__ == '__main__':
    user = User(name='Pham Dinh Khanh', age=27, gender='male', occupation='AI Engineer')
    item = Item(item_id='123', item_name='keo vuốt tóc', item_price=50.000)
    order = Order(user=user, item=item, item_quant=2)
    total_cost = order.cost()
    print(total_cost)
