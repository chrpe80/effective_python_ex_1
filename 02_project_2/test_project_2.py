import unittest
from unittest.mock import patch
from datetime import datetime
import random
from project_2 import Player, GamePlay


class TestPlayer(unittest.TestCase):
    def test_valid_name_and_input(self):
        """Tests if valid name and birthdate are set correctly"""
        instance = Player("Christian Persson", "19800621")
        self.assertEqual(instance.name, "Christian Persson")
        self.assertEqual(instance.birthdate, datetime.strptime("19800621", "%Y%m%d"))

    def test_invalid_name(self):
        """Tests if invalid name is handled correctly."""
        instance = Player("ChristianPersson", "19800621")
        self.assertEqual(instance.name, None)

    def test_invalid_birthdate(self):
        """Tests if invalid birthdate is handled correctly"""
        instance = Player("Christian Persson", "1980")
        self.assertEqual(instance.birthdate, None)

    def test_validate_name_valid(self):
        """Tests if the validate_name static method works with valid input"""
        self.assertEqual(Player.validate_name("Christian P"), "Christian P")

    def test_validate_name_invalid(self):
        """Tests if the validate_name static method works with invalid input"""
        self.assertEqual(Player.validate_name("ChristianP"), None)

    def test_validate_birthdate_valid(self):
        """Tests if the validate_birthdate static method works with valid input"""
        self.assertEqual(Player.validate_birthdate("19800621"), datetime.strptime("19800621", "%Y%m%d"))

    def test_validate_birthdate_invalid(self):
        """	Tests if the validate_birthdate static method works with invalid input"""
        self.assertEqual(Player.validate_birthdate("1980"), None)

    def test_age_had_birthday(self):
        """	Tests if the age property works when the player has already had their birthday"""
        instance = Player("Christian Persson", "19800621")
        self.assertEqual(instance.age, 43)

    def test_age_not_had_birthday(self):
        """	Tests if the age property works when the player has not had their birthday yet"""
        instance = Player("Christian Persson", "19801221")
        self.assertEqual(instance.age, 42)

    def test_if_of_age_old_enough(self):
        """	Tests if the is_of_age property returns True for an old enough player"""
        instance = Player("Christian Persson", "19800621")
        self.assertEqual(instance.is_of_age, True)

    def test_if_of_age_not_old_enough(self):
        """	Tests if the is_of_age property returns False for a too young player"""
        instance = Player("Christian Persson", "20200621")
        self.assertEqual(instance.is_of_age, False)

    def test_ready_to_go_name_is_none(self):
        """Tests if the ready_to_go method returns False when the name is invalid"""
        instance = Player("ChristianP", "19800621")
        with patch("builtins.print") as mocked_print:
            instance.ready_to_go()
            mocked_print.assert_called_with("Incomplete input")
        self.assertEqual(instance.ready_to_go(), False)

    def test_ready_to_go_not_none_of_age(self):
        """Tests if the ready_to_go method returns True when the player is valid and of age"""
        instance = Player("Christian Persson", "19800621")
        self.assertEqual(instance.ready_to_go(), True)

    def test_ready_to_go_not_none_not_of_age(self):
        """Tests if the ready_to_go method returns False when the player is valid but not of age."""
        instance = Player("Christian Persson", "20200621")
        with patch("builtins.print") as mocked_print:
            instance.ready_to_go()
            mocked_print.assert_called_with("Sorry, you are too young to play this game")
        self.assertEqual(instance.ready_to_go(), False)


class TestGamePlay(unittest.TestCase):
    def test_init(self):
        """Tests if GamePlay instance is initialized with zero attempts"""
        instance = GamePlay()
        self.assertEqual(instance._attempts, 0)

    def test_get_player_name(self):
        """Tests if get_player_name method correctly gets player name from input"""
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "Christian Persson"
            self.assertEqual(GamePlay.get_player_name(), "Christian Persson")

    def test_get_birthdate(self):
        """Tests if get_birthdate method correctly gets birthdate from input"""
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "19800621"
            self.assertEqual(GamePlay.get_player_name(), "19800621")

    def test_lucky_list(self):
        """Tests if lucky_list setter works correctly"""
        instance = GamePlay()
        instance.lucky_list = random.sample(range(1, 100), 9)
        self.assertEqual(len(instance._lucky_list), 9)

    def test_lucky_number(self):
        """Tests if lucky_number setter works correctly"""
        instance = GamePlay()
        instance.lucky_number = random.randint(1, 100)
        self.assertIn(instance._lucky_number, range(1, 100))

    def test_guess(self):
        """Tests if guess setter works correctly"""
        instance = GamePlay()
        instance._guess = 80
        self.assertEqual(instance._guess, 80)

    def test_check_if_guess_is_correct(self):
        """Tests if check_if_guess_is_correct method works as expected"""
        instance = GamePlay()
        instance._guess = 2
        instance._lucky_number = 2
        expectation = True
        snippet = instance.check_if_guess_is_correct()
        self.assertEqual(snippet, expectation)

    def test_get_guess(self):
        """Tests if get_guess method correctly gets guess from input"""
        with patch("builtins.input") as mocked_input:
            mocked_input.return_value = "2"
            self.assertEqual(GamePlay.get_guess(), 2)

    def test_print_message(self):
        """	Tests if print_message correctly prints the required message"""
        with patch("builtins.print") as mocked_print:
            GamePlay.print_message()
            mocked_print.assert_called_with("Your guess has to be an integer between 0-100")


if __name__ == "__main__":
    unittest.main()
