class User:
  nation = 'VietNam'
  def __init__(self, name, age, gender, occupation):
    self.name = name
    self.age = age
    self.gender = gender
    self.occupation = occupation

  # Các hàm buy, search là những hàm phương thức.
  def buy(self, item):
    print('you bought {}'.format(item))
  
  def search(self, term):
    print('you search: {}'.format(term))
