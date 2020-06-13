file  = open("data/skychallenge_skyphrase_input.txt", "r")
lines = file.readlines()
valid = 0 #valid skyphrases

for line in lines:
    words = line.split(" ")
    
    if len(set(words)) == len(words):
        valid += 1


print(f"Valid skyphrases: {valid}")