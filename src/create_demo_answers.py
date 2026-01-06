import csv
import random

filename = "./demo_data/answer_key.csv"

with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Section", "Question", "Sub-question", "Answer"])

    for i in range(1, 41):
        answer = random.choice(["A", "B", "C", "D"])
        writer.writerow(["I", i, "", answer])

    for i in range(1, 9):
        for sub in ["a", "b", "c", "d"]:
            answer = random.choice(["True", "False"])
            writer.writerow(["II", i, sub, answer])

    list_answer = ["-0.78", "1365", "24.6", "-0.8", "9525" ,"3333"]
    for i in range(len(list_answer)):
        writer.writerow(["III", i+1, "", list_answer[i]])

print(f"File '{filename}' created")