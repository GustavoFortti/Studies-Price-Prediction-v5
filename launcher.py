from sys import argv
from os import system
from time import sleep
from datetime import datetime
from src.Iterator.bot import Interator

def main():
    args = argv[1:]

    if (args[0] == '-pr'):
        system('python3 ./src/main.py 0 pred')
        # while (True):
        #     print(datetime.now())
        #     if ((datetime.now().strftime("%M") == "59") & (datetime.now().strftime("%S") in ["30", "31", "32"])):
        #         system('python3 ./src/main.py 0 pred')
        #         bot = Interator()
        #         bot.run()
        #         sleep((60*58))
        #     sleep(1)
    elif (args[0] == '-tr'):
        system('python3 ./src/main.py 0 train')
    elif (args[0] == '-td-gen'):
        system('python3 ./src/main.py 0 gen')
    elif (args[0] == '-td'):
        system('python3 ./src/main.py 0 die')
    elif (args[0] == '-t0'):
        system('python3 ./src/main.py 0 test')
    elif (args[0] == '-tn'):
        for i in range(890, int(args[1])):
            print("index " + str(i))
            system('python3 ./src/main.py ' + str(i) + ' test')

if __name__ == "__main__":
    # bot = Interator()
    # bot.run()
    main()
   