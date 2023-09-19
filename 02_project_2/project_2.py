from datetime import datetime
import random


class Player:
    def __init__(self, name=None, birthdate=None):
        self.name = self.validate_name(name)
        self.birthdate = self.validate_birthdate(birthdate)

    @staticmethod
    def validate_name(name: str):
        """
        Validates the given name to ensure it's alphabetic and contains one space
        :rtype: str or None
        """
        nr_of_spaces = name.count(" ")
        is_alpha = all(x.isalpha() for x in name.split(" "))
        if nr_of_spaces == 1 and is_alpha:
            return name
        else:
            return None

    @staticmethod
    def validate_birthdate(birthdate: str):
        """
        Validates and converts the given birthdate to a datetime object
        :rtype: datetime or None
        """
        try:
            return datetime.strptime(str(birthdate).replace("-", ""), '%Y%m%d')
        except ValueError:
            return None

    @property
    def age(self):
        """
        Calculates the age of the player based on their birthdate
        :rtype: int or None
        """
        if self.birthdate is None:
            return None

        today = datetime.today()
        age = today.year - self.birthdate.year

        if (today.month, today.day) >= (self.birthdate.month, self.birthdate.day):
            return age

        return age - 1

    @property
    def is_of_age(self):
        """
        Checks if the player is 18 years old or older
        :rtype: bool
        """
        if self.age >= 18:
            return True
        return False

    def ready_to_go(self):
        """
        Checks if the player is 18 years old or older
        :rtype: bool
        """
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
        """
        Gets the player's name via input
        :rtype: str
        """
        name = input("Enter your name: ")
        return name

    @staticmethod
    def get_birthdate():
        """
        Gets the player's birthdate via input
        :rtype: str
        """
        birthdate = input("Enter your birthdate (yyyymmdd): ")
        return birthdate

    @property
    def lucky_list(self):
        """
        Gets the lucky_list
        :rtype: list
        """
        return self._lucky_list

    @lucky_list.setter
    def lucky_list(self, value):
        """
        Sets the lucky list
        :rtype: None
        """
        self._lucky_list = value

    @property
    def lucky_number(self):
        """
        Gets the lucky number
        :rtype: int
        """
        return self._lucky_number

    @lucky_number.setter
    def lucky_number(self, value):
        """
        Sets the lucky number
        :rtype: None
        """
        self._lucky_number = value

    @property
    def guess(self):
        """
        Gets the player's guess
        :rtype: int
        """
        return self._guess

    @guess.setter
    def guess(self, value):
        """
        Sets the player's guess if it's an integer between 0-100
        :rtype: None
        """
        if 0 <= int(value) <= 100:
            self._guess = value

    def check_if_guess_is_correct(self):
        """
        Checks if the player's guess matches the lucky number
        :rtype: bool
        """
        if self._guess == self._lucky_number:
            return True

        return False

    @staticmethod
    def get_guess():
        """
        Gets the player's guess via input and validates it
        :rtype: int or None
        """
        guess = input("Enter guess: ")
        if guess.isnumeric() and 0 <= int(guess) <= 100:
            return int(guess)
        return None

    @staticmethod
    def print_message():
        """
        Prints a message stating that the guess must be an integer between 0-100
        :rtype: None
        """
        print("Your guess has to be an integer between 0-100")

    def start_game(self):
        """
        Starts the game loop, allowing the player to guess many times
        :rtype: None
        """
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
        """
        Runs the game
        :rtype: None
        """
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
