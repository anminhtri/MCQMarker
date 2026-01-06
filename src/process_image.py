import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pandas as pd
import cv2
import imutils
from math import ceil
import numpy as np
from collections import defaultdict, OrderedDict
from src.model import CNN_Model


def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def merge_overlapping_contours(contours):
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    merged_boxes = []

    for box in bounding_boxes:
        x, y, w, h = box
        merged = False
        for idx, merged_box in enumerate(merged_boxes):
            x_m, y_m, w_m, h_m = merged_box
            if not (x > x_m + w_m or x + w < x_m or y > y_m + h_m or y + h < y_m):  # Check overlap
                new_x = min(x, x_m)
                new_y = min(y, y_m)
                new_w = max(x + w, x_m + w_m) - new_x
                new_h = max(y + h, y_m + h_m) - new_y
                merged_boxes[idx] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        if not merged:
            merged_boxes.append((x, y, w, h))

    return merged_boxes

def crop_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    img_canny = cv2.Canny(blurred, 10, 50)

    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    output = img.copy()
    cv2.drawContours(output, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite("./results/all_contours.jpg", output)
    
    merged_boxes = merge_overlapping_contours(cnts)

    count = 0
    expected_destination = ""
    ans_blocks_pI = []
    ans_blocks_pII = []
    ans_blocks_pIII = []
    ans_blocks_sbd = []
    cau = 78
    p1 = 3
    for i, (x, y, w, h) in enumerate(merged_boxes):
        if  w * h > 1000:  
            count += 1
            cropped = img[y:y + h, x:x + w]
            if count < 2: 
                expected_destination = "./preprocessed_data/Part_III.jpg"
                ans_blocks_pIII.append(gray_img[y:y + h, x:x + w])
            elif count < 6:
                expected_destination = f"./preprocessed_data/Part_II_section_{cau}.jpg"
                ans_blocks_pII.append(gray_img[y:y + h, x:x + w])
                cau -=22
            elif count < 10 :
                expected_destination = f"./preprocessed_data/Part_I_section_{p1}.jpg"
                ans_blocks_pI.append(gray_img[y:y + h, x:x + w])
                p1 -= 1
            else: 
                expected_destination = "./preprocessed_data/Part_0.jpg"
                ans_blocks_sbd.append(gray_img[y:y + h, x:x + w])
            cv2.imwrite(expected_destination, cropped)
            print(f"Saved cropped_{i}.jpg with size {w}x{h} and area {w * h}")

    ans_blocks_pI = ans_blocks_pI[::-1]
    ans_blocks_pII = ans_blocks_pII[::-1]
    return ans_blocks_pI, ans_blocks_pII, ans_blocks_pIII, ans_blocks_sbd

        
def process_ans_blocks(ans_blocks, part=1):
    list_answers = []
    if part == 1:
        x = 0   
        for i, pic in enumerate(ans_blocks):
            x = 0
            new_pic = pic[ 20 : , : ]
            h = new_pic.shape[0]
            w = new_pic.shape[1]
            h_new = h // 10
            for j in range(10):
                cropped = new_pic[x: x + h_new, :]
                x += h_new
                list_answers.append(cropped)

    elif part == 2:
        x = 0
        for i, pic in enumerate(ans_blocks):
            x = 0
            new_pic = pic[ 27 : , : ]
            h = new_pic.shape[0]
            w = new_pic.shape[1]
            h_new = h // 4
            for j in range(4):
                cropped = new_pic[x: x + h_new, :]
                x += h_new
                list_answers.append(cropped)

    elif part == 3:
        x = 0
        y = 0
        for i, pic in enumerate(ans_blocks):
            w = pic.shape[1]
            new_w = w // 6
            x = 0
            y = 0
            for j in range(6):
                new_pic = pic[ 32: , y: y + new_w]
                new_pic = new_pic[: , 22:]
                w_new = new_pic.shape[1] // 4
                y2 = 0
                for k in range(4):
                    cropped = new_pic[:, y2:y2 + w_new]
                    y2 += w_new
                    list_answers.append(cropped)

                y += new_w
    elif part == 4:
        pic = ans_blocks[0]
        new_pic = pic[24: , 345 :]
        pic_sbd = new_pic[: , 20: new_pic.shape[1] - 76]
        w_new = pic_sbd.shape[1] // 6
        y = 0
        for i in range(6):
            cropped = pic_sbd[:, y: y + w_new]
            y += w_new
            list_answers.append(cropped)

    else:
        pic = ans_blocks[0]
        new_pic = pic[24: , 345 :]
        made_pic = new_pic[: , new_pic.shape[1]-56: new_pic.shape[1]-2]
        w_new = made_pic.shape[1] // 3
        y = 0
        for i in range(3):
            cropped = made_pic[: , y: y + w_new]
            y += w_new
            list_answers.append(cropped)

    return list_answers
def process_list_ans(list_answers, part = 1):
    if part == 1:
        buble_choice = []
        for i, pic in enumerate(list_answers):
            new_pic = pic[: , 24:]
            new_width = new_pic.shape[1] // 4
            y = 0
            for j in range(4):
                choice = new_pic[ : , y: y + new_width]
                choice = cv2.threshold(choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                choice = cv2.resize(choice, (28, 28), interpolation= cv2.INTER_AREA)
                choice = choice.reshape((28, 28, 1))
                y += new_width
                buble_choice.append(choice)

    elif part == 2:
        buble_choice = []
        for i, pic in enumerate(list_answers):
            new_pic = pic[: , 24:]
            new_width = new_pic.shape[1] // 4
            y = 0
            for j in range(4):
                choice = new_pic[ : , y: y + new_width]
                choice = cv2.threshold(choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                choice = cv2.resize(choice, (28, 28), interpolation= cv2.INTER_AREA)
                choice = choice.reshape((28, 28, 1))
                y += new_width
                buble_choice.append(choice)
    elif part == 3:
        buble_choice = []
        for i, pic in enumerate(list_answers):
            new_pic = pic
            new_height = new_pic.shape[0] // 12
            x = 0
            for j in range(12):
                choice = new_pic[ x : x + new_height, :]
                choice = cv2.threshold(choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                choice = cv2.resize(choice, (28, 28), interpolation= cv2.INTER_AREA)
                choice = choice.reshape((28, 28, 1))
                x += new_height
                buble_choice.append(choice)

    else:
        buble_choice = []
        for i, pic in enumerate(list_answers):
            new_pic = pic
            new_height = new_pic.shape[0] // 10
            x = 0
            for j in range(10):
                choice = new_pic[ x : x + new_height, :]
                choice = cv2.threshold(choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                choice = cv2.resize(choice, (28, 28), interpolation= cv2.INTER_AREA)
                choice = choice.reshape((28, 28, 1))
                x += new_height
                buble_choice.append(choice)

    return buble_choice
def map_answer_p1(idx):
    if idx % 4 == 0:
        answer_circle = 'A'
    elif idx % 4 == 1:
        answer_circle = 'B'
    elif idx % 4 == 2:
        answer_circle = 'C'
    else:
        answer_circle = 'D'
    return answer_circle

def map_answer_p2(idx):
    if idx %2 == 0:
        answer_circle = True
    else:
        answer_circle = False
    return answer_circle

def map_answer_p3(idx):
    mapping_answer_circle = "-.0123456789"
    return mapping_answer_circle[idx%12]
def map_answer_sbd_and_made(idx):
    mapping_answer_circle = "0123456789"
    return mapping_answer_circle[idx%10]
def get_answers(list_answers, part = 1):
    if part == 1:
        results = defaultdict(list)
        model = CNN_Model('./src/weight.h5').build_model(rt=True)
        list_answers = np.array(list_answers)
        scores = model.predict_on_batch(list_answers / 255.0)
        for idx, score in enumerate(scores):
            question = idx // 4

            if score[1] > 0.9:  
                chosed_answer = map_answer_p1(idx)
                results[question + 1].append(chosed_answer)
        return results
    elif part == 2:
        results = defaultdict(list)
        model = CNN_Model('./src/weight.h5').build_model(rt=True)
        list_answers = np.array(list_answers)
        scores = model.predict_on_batch(list_answers / 255.0)
        
        for idx, score in enumerate(scores):
            question = idx // 4
            if score[1] > 0.9: 
                chosed_answer = map_answer_p2(idx)
                results[question + 1].append(chosed_answer)

        odd_questions, even_questions = [], []
        for key, val in results.items():
            odd_questions.append(val[0])
            even_questions.append(val[1])

        new_results = defaultdict(lambda: defaultdict(list))

        number, idx = 1, 0
        for i in range(len(odd_questions)):
            idx += 1
            if idx == 5:
                idx = 1
                number += 2
            type_question = ['a', 'b', 'c', 'd'][idx - 1]
            new_results[number][type_question].append(odd_questions[i])

        number, idx = 2, 0
        for i in range(len(even_questions)):
            idx += 1
            if idx == 5:
                idx = 1
                number += 2
            type_question = ['a', 'b', 'c', 'd'][idx - 1]
            new_results[number][type_question].append(even_questions[i])

        sorted_results = OrderedDict(sorted(new_results.items(), key=lambda x: x[0]))

        return sorted_results
    elif part == 3:
        results = defaultdict(list)
        model = CNN_Model('./src/weight.h5').build_model(rt=True)
        list_answers = np.array(list_answers)
        scores = model.predict_on_batch(list_answers / 255.0)
        for idx, score in enumerate(scores):
            question = idx // 12

            if score[1] > 0.9: 
                chosed_answer = map_answer_p3(idx)
                results[question + 1].append(chosed_answer)
        start, end = 1, 5
        new_results = defaultdict(list)
        ans = ""
        for i in range(1, 7):
            ans=""
            for j in range(start, end):
                ans = ans + results[j][0]
            new_results[i].append(ans)
            start += 4
            end += 4
        return new_results
    elif part == 4:
        results = defaultdict(list)
        model = CNN_Model('./src/weight.h5').build_model(rt=True)
        list_answers = np.array(list_answers)
        scores = model.predict_on_batch(list_answers / 255.0)
        for idx, score in enumerate(scores):
            question = idx // 10

            if score[1] > 0.9: 
                chosed_answer = map_answer_sbd_and_made(idx)
                results[question + 1].append(chosed_answer)

        return results
        
def get_full_answers(link_img):
    img = cv2.imread(link_img)                                                                                                                                                                                                                                      
    cropped_blocksI, cropped_blocksII, cropped_blocksIII, cropped_blocks_sbd = crop_image(img)
    list_answers_pI = process_ans_blocks(cropped_blocksI, 1)
    list_answers_pII = process_ans_blocks(cropped_blocksII, 2)
    list_answers_pIII = process_ans_blocks(cropped_blocksIII, 3)
    list_answers_sbd = process_ans_blocks(cropped_blocks_sbd, 4)
    list_answers_made = process_ans_blocks(cropped_blocks_sbd, 5)
    
    list_answers_pI =process_list_ans(list_answers_pI, 1)
    list_answers_pII = process_list_ans(list_answers_pII, 2)
    list_answers_pIII = process_list_ans(list_answers_pIII, 3)
    list_answers_sbd = process_list_ans(list_answers_sbd, 4)
    list_answers_made = process_list_ans(list_answers_made, 4)
    
    answersI = get_answers(list_answers_pI, 1)
    answersII = get_answers(list_answers_pII, 2)
    answersIII = get_answers(list_answers_pIII, 3)
    answerssbd = get_answers(list_answers_sbd, 4)
    answersmade = get_answers(list_answers_made, 4)
    print(answersI)
    print(answersII)
    print(answersIII)

    return answersI, answersII, answersIII, answerssbd, answersmade


def grade_part1(df, student_answer_dict):
    score = 0
    answer_dict = defaultdict(list)
    for _, row in df.iterrows():
        question_number = int(row["Question"])
        part = row["Section"]
        if 1 <= question_number <= 40 and part =="I":
            answer_dict[question_number].append(row["Answer"])
    for key, value in answer_dict.items():
        if value == student_answer_dict[key]:
            score+=1
    return f"{score}/40"

def grade_part2(df, student_answer_dict):
    score = 0
    answer_dict = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        question_number = int(row["Question"])
        part = row["Section"]
        if 1 <= question_number <= 8 and part == "II":
            answer_dict[question_number][row["Sub-question"]].append(row["Answer"])
    for key, value in answer_dict.items():
        for sub_key, correct_ans in value.items():
            if correct_ans == list(map(str, student_answer_dict[key][sub_key])):
                score += 1
    return f"{score}/32"

def grade_part3(df, student_answer_dict):
    score = 0
    answer_dict = defaultdict(list)
    for _, row in df.iterrows():
        question_number = int(row["Question"])
        part = row["Section"]
        if 1 <= question_number <= 6 and part == "III":
            answer_dict[question_number].append(row["Answer"])
    
    for key, value in answer_dict.items():  
        if value == student_answer_dict.get(key, []):  
            score += 1
    return f"{score}/6"
    
def main():
    p1, p2, p3, sbd, made = get_full_answers("./demo_data/form2.png")
    df = pd.read_csv("./demo_data/answer_key.csv")
    diemp1 = grade_part1(df, p1)
    diemp2 = grade_part2(df, p2)
    diemp3 = grade_part3(df, p3)
    sbd_str = ''.join([''.join(val) for val in sbd.values()])

    made_str = ''.join([''.join(val) for val in made.values()])

    img = cv2.imread("./demo_data/form2.png")
    cv2.putText(img, diemp1, (400, 280), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, diemp2, (400, 490), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, diemp3, (400, 610), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sbd_str, (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, made_str, (540, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite("./results/score.jpg", img)

    cv2.imshow("Your score", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()