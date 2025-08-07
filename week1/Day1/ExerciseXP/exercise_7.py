#Exercise 7 : Odd or Even
#Instructions
#Write code that asks the user for a number and determines whether this number is odd or even.
user_input = input("Enter a number:")
try:
    number = int(user_input)
    if number % 2 == 0:
        print(f"{number} is an even number")
    elif number % 3 == 0:
        print(f"{number} is an odd number")
    else:
        print(f"{number} is a prime number")

except:
    print(f"{user_input} is not a number")
