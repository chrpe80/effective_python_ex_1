from datetime import datetime
import random


class Player:
    def __init__(self, name=None, birthdate=None):
        self.name = self.validate_name(name)
        self.birthdate = self.validate_birthdate(birthdate)

    @staticmethod
    def validate_name(name: str):
        nr_of_spaces = name.count(" ")
        is_alpha = all(x.isalpha() for x in name.split(" "))
        if nr_of_spaces == 1 and is_alpha:
            return name
        else:
            return None

    @staticmethod
    def validate_birthdate(birthdate: str):
        try:
            return datetime.strptime(str(birthdate).replace("-", ""), '%Y%m%d')
        except ValueError:
            return None

    @property
    def age(self):
        if self.birthdate is None:
            return None

        today = datetime.today()
        age = today.year - self.birthdate.year

        if (today.month, today.day) >= (self.birthdate.month, self.birthdate.day):
            return age

        return age - 1

    @property
    def is_of_age(self):
        if self.age >= 18:
            return True
        return False

    def ready_to_go(self):
        if self.name is None or self.birthdate is None:
            print("Incomplete input")
            return False
        elif self.name is not None and self.birthdate is not None and self.is_of_age:
            return True
        else:
            print("Sorry, you are too young to play this game")
            return False


class GamePlay:
    def __init__(self):
        self._attempts = 0
        self._lucky_list = []
        self._lucky_number = 0
        self._guess = 0

        name = self.get_player_name()
        birthdate = self.get_birthdate()
        self.player = Player(name, birthdate)
        if self.player.ready_to_go():
            self.start_game()
        else:
            pass

    @staticmethod
    def get_player_name():
        name = input("Enter your name: ")
        return name

    @staticmethod
    def get_birthdate():
        birthdate = input("Enter your birthdate (yyyymmdd): ")
        return birthdate

    @property
    def lucky_list(self):
        return self._lucky_list

    @lucky_list.setter
    def lucky_list(self, value):
        self._lucky_list = value

    @property
    def lucky_number(self):
        return self._lucky_number

    @lucky_number.setter
    def lucky_number(self, value):
        self._lucky_number = value

    @property
    def guess(self):
        return self._guess

    @guess.setter
    def guess(self, value):
        if 0 <= int(value) <= 100:
            self._guess = value

    def check_if_guess_is_correct(self):
        if self._guess == self._lucky_number:
            return True

        return False

    @staticmethod
    def get_guess():
        guess = input("Enter guess: ")
        if guess.isnumeric() and 0 <= int(guess) <= 100:
            return int(guess)
        return None

    @staticmethod
    def print_message():
        print("Your guess has to be an integer between 0-100")

    def start_game(self):
        while True:
            self.run()
            while True:
                answer = input("Play again [y/n]: ").lower()
                if answer == "y":
                    self._attempts = 0
                    break
                elif answer == "n":
                    print("Ok, bye!")
                    return
                else:
                    print("Valid inputs are [y/n]")

    def run(self):
        self.lucky_list = random.sample(range(1, 100), 9)
        self.lucky_number = random.randint(1, 100)
        self._lucky_list.append(self._lucky_number)
        print(self._lucky_list)

        while True:
            guess = self.get_guess()
            if guess is not None:
                self.guess = int(guess)
                self._attempts += 1
                check = self.check_if_guess_is_correct()
                if check:
                    print(f"You guessed correctly on the {self._attempts} try!")
                    break
                else:
                    minimum = self._lucky_number - 10
                    maximum = self._lucky_number + 10
                    self.lucky_list = [nr for nr in self._lucky_list if minimum <= nr <= maximum]
                    print(self._lucky_list)
            else:
                self.print_message()


# GamePlay()
