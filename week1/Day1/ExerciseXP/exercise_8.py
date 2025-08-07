#Exercise 8 : What’s your name ?
#Instructions
#Write code that asks the user for their name and determines whether or not you have the same name. Print out a funny message based on the outcome.
voldermort = "Tom"
user_name = input("What is your name:")
if voldermort == user_name:
    print(f"Two {voldermort} can not exist")
else:
    print(f"Hello {user_name}! My name is {voldermort}")
