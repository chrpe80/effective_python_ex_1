# Projekt 2: Computer game: What is your lucky number?

A game between computer and human (min 18 years old).

## Important: Kindly note

**This project will be**: 

- In a virtual environment
- In a git repo
- In a GitHub private repo with requirements.txt and .gitignore
- Using OOP only
- Tested

## Game steps and rules:

1. Input: player’s full name -> "player_name"
   * The variable player_name should be a string that is alphabetic and one whitespace only between first name and last name
2. Input: player's birthdate (yyyymmdd). 
   * Validate the input. If it's ok -> "player_birthdate"
3. The program calculates the players age from the player_birthdate -> "player_age"
4. The program compares if player_age >= to 18 (current year is 2023). If True, then we move to step #5, if not, start again from step #2

**Once we have the right player_name and age**:

5. The program generates a list of 9 integers between 0-100 -> "lucky_list"
6. The program generates a lucky number between 0-100 -> "lucky_number", and adds it to the lucky_list so lucky list will have 10 integers now
7. The computer prints the lucky_list and asks the player to pick the lucky number from the lucky list -> "player_input"
8. If the player chooses the lucky number, then the program prints: “Congrats, the game is over! And you got the lucky number from {attempts} attempts ”, where "attempts" is a variable that keeps track of how many times the player tried to get the lucky number
9. Then the program prints to the player: “Would you like to play again? (Input y: Yes, or n:NO)
   * if yes, we go back to step #5, if no, we exit
10. If wrong guess, the program generates a new list that contains numbers from the lucky_list and has only numbers that differ 10 from the lucky number -> "shorter_lucky_list"

## Example:

```python
lucky_list = [5, 1, 20, 99, 70, 12, 22, 2, 89, 15]

Lucky_number = 12

player_input = 70

# Shorter_lucky_list will be between:

min = 12 - 10
max = 12 + 10

shorter_lucky_list = [5,20,12,22,2,15]
```



11. The program prints: “this is try {attempt} and the new list is: {shorter_lucky_list}, choose the lucky number?"
    1. If the player chooses the lucky number, go back to step #8
    2. Otherwise, the program deletes the wrong number that the player entered in step#10, from the shorter lucky list and back again to step #10.
    3. The game ends when the player guesses the lucky number or shorter_lucky_list has just 2 integers left!