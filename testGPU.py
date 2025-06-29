




# x = int(input("Enter a number: "))

# print( x * 2 ) 


# def add(x, y):
#     return x + y

# def subtract(x, y):
#     return x - y

# def multiply(x, y):
#     return x * y

# def divide(x, y):
#     if y == 0:
#         raise ZeroDivisionError("Cannot divide by zero")
#     return x / y

# def calculator():
#     print("Welcome to the Python Calculator (type 'exit' to quit)")
#     while True:
#         op = input("Choose operation (+, -, *, /): ")
#         if op.lower() == 'exit':
#             print("Goodbye!")
#             break
#         if op not in ('+', '-', '*', '/'):
#             print("Invalid operation. Please choose +, -, *, or /.")
#             continue

#         try:
#             num1 = float(input("Enter first number: "))
#             num2 = float(input("Enter second number: "))
#         except ValueError:
#             print("Invalid number entered. Try again.")
#             continue

#         try:
#             if op == '+':
#                 result = add(num1, num2)
#             elif op == '-':
#                 result = subtract(num1, num2)
#             elif op == '*':
#                 result = multiply(num1, num2)
#             else:  # op == '/'
#                 result = divide(num1, num2)
#             print(f"Result: {num1} {op} {num2} = {result}")
#         except ZeroDivisionError as e:
#             print(e)

# if __name__ == "__main__":
#     calculator()

# int 56

# x = int(input()) số nguyên 

# float 45.3454 

# y = float(input()) số thập phân 

# Kiểu dữ liệu string 
# Syntax: str

# s = str(input()) 

# string chuỗi 
# "dsfhgkj dfkjshdkjh sfdgjshkdf 3458dsfdas 78945hdkjsf 8734985 dshskjh"
# "dsgsfdd fsdfg 324 324 234 324"
# 'sdf jfaksjh fasdf'
# khoảng chắn cũng là 1 kí tự 

# Nhập số 4 với số 5 
x = int(input("Nhập số thứ nhất: ")) # x = 4 
y = int(input("Số thứ hai phải nhỏ hơn số thứ nhất. Nhập số thứ hai đi: ")) # x = 4 , y = 5  

# x = 4 , y = 5
x = y + 10 # x = 5 , y = 5 
#   = 5 + 10 = 15 
print("In ra số mới vừa nhập:" , x )
print("In ra số mới vừa nhập:" , y )

# print("Nhập dấu (+, -, *, /): ") 
# dau = str(input()) 



# print("Kết quả phép nhân: ", x * y ) # SHIFT + 8 
# print("Kết quả phép cộng: ", x + y ) # SHIFT + 9
# print("Kết quả phép trừ: ", x - y ) # SHIFT + 7
# print("Kết quả phép chia: ", x // y ) # SHIFT + 6
# print("Kết quả phép chia: ", x / y ) # SHIFT + 6



# / : chia lấy cả thập ví dụ : 234.54343 . VD 5 / 2 = 2.5   
# // : chia lấy nguyên ví dụ : 234.54343 // 10 = 23 , VD 5 // 2 = 2

# print(type(x)) # type(x) sẽ trả về kiểu dữ liệu của biến x 

# Tú : nguyên 
# Huy : nguyên , Khoa: thập phân 

# print("In ra số mới vừa nhập:" , x , "và kiểu dữ liệu của nó là:", type(x) )
# print("In ra số mới vừa nhập:" , y , "và kiểu dữ liệu của nó là:", type(y) )

