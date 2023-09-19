import unittest
from unittest.mock import patch, PropertyMock
from project_2 import GamePlay


class TestSystem(unittest.TestCase):
    def test_game_correct_guess(self):
        """	Tests if the game ends properly when the user guesses correctly and chooses not to play again"""
        with patch("builtins.input") as mocked_input, patch("builtins.print") as mocked_print:
            mocked_input.side_effect = ["Christian P", "19800621", "0", "n"]
            with patch("project_2.GamePlay.lucky_number", new_callable=PropertyMock):
                GamePlay()
                mocked_print.assert_called_with("Ok, bye!")

    def test_game_correct_guess_start_again(self):
        """Tests if the game restarts when the user guesses correctly and chooses to play again"""
        with patch("builtins.input") as mocked_input, patch("builtins.print") as mocked_print:
            mocked_input.side_effect = ["Christian P", "19800621", "0", "y", "0", "n"]
            with patch("project_2.GamePlay.lucky_number", new_callable=PropertyMock):
                instance = GamePlay()
                self.assertEqual(instance._attempts, 1)
                mocked_print.assert_any_call(f"You guessed correctly on the 1 try!")

    def test_game_wrong_guess(self):
        """Tests if the game proceeds correctly when the user guesses wrong initially but corrects it afterward"""
        with patch("builtins.input") as mocked_input, patch("builtins.print") as mocked_print:
            mocked_input.side_effect = ["Christian P", "19800621", "1", "0", "n"]
            with patch("project_2.GamePlay.lucky_number", new_callable=PropertyMock):
                instance = GamePlay()
                self.assertEqual(instance._attempts, 2)
                mocked_print.assert_any_call(f"You guessed correctly on the 2 try!")

    def test_game_invalid_guess(self):
        """Tests if the game handles an invalid guess input and provides the user with a correct message"""
        with patch("builtins.input") as mocked_input, patch("builtins.print") as mocked_print:
            mocked_input.side_effect = ["Christian P", "19800621", "invalid", "0", "n"]
            with patch("project_2.GamePlay.lucky_number", new_callable=PropertyMock):
                GamePlay()
                mocked_print.assert_any_call("Your guess has to be an integer between 0-100")


if __name__ == "__main__":
    unittest.main()
