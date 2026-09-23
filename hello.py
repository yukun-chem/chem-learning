def main():
    hello()
    name = input("what's your name? ").strip().title()
    first, last = name.split(" ")
    hello(last)

def hello(to = "world"):
    print(f"hello, {to}")

main()
 