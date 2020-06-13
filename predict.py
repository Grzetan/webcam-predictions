import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pygame as pygame
import numpy as np
import sys
from model import Network
from prediction_functions import arr

def main(model_path,model_index):
    torch.set_grad_enabled(False)

    pygame.init()

    W, H = 640, 700
    WIN = pygame.display.set_mode((W,H))
    pygame.display.set_caption('Predicting from ' + model_path)
    CLOCK = pygame.time.Clock()
    FPS = 30

    run = True
    FONT = pygame.font.SysFont('comicsans', 200)
    video = cv2.VideoCapture(0)
    status = 'waiting'
    model = torch.load(model_path)

    def events():
        nonlocal run
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                sys.exit()

    def create_button(text, color, highlight_color, x, y, width, height, action=None):
        BTN_FONT = pygame.font.SysFont('comicsans', 40)

        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if mouse[0] > x and mouse[0] < x + width and mouse[1] > y and mouse[1] < y + height:  
            if click[0] == 1 and action != None:
                action()
            btn_color = highlight_color
        else:
            btn_color = color
        btn_text = BTN_FONT.render(text, 1, (0,0,0))

        if x is False:
            x = W//2-width//2

        pygame.draw.rect(WIN, btn_color, (x,y,width,height))
        WIN.blit(btn_text, (x+width//2 - btn_text.get_width() // 2, y+height//2 - btn_text.get_height() // 2))

    def go_back():
        video.release()
        import main
        main.main()

    def refresh_win(frame, prediction):
        WIN.fill(0)

        #live webcam

        WIN.blit(frame, (0,0))

        #display prediction

        img = f'{np.argmax(prediction)+1}'
        
        TEXT = FONT.render(img, 1, (255,0,0))
        WIN.blit(TEXT, (W//2 - TEXT.get_width()//2, H//2-TEXT.get_height()//2))

        #go back

        create_button('<--', (128,128,128), (200,200,200), 30,30,50,50, action=go_back)

        pygame.display.update()

    while run:
        CLOCK.tick(FPS)
        
        #get frame

        check, frame = video.read()
        display_frame = pygame.transform.rotate(pygame.surfarray.make_surface(frame), -90)

        predict_frame = cv2.resize(frame, (160,120))
        predict_frame = cv2.cvtColor(predict_frame, cv2.COLOR_BGR2GRAY)
        predict_frame = torch.tensor(predict_frame).float()
        predict_frame = predict_frame.unsqueeze(0)
        predict_frame = predict_frame.unsqueeze(0)
        prediction = model(predict_frame)[0]
        prediction_index = int(np.argmax(prediction))
        
        arr[model_index](prediction_index)

        # if prediction_index == 0:
        #     #Do something if the prediction is 0
        # elif prediction_index == 1:
        #     #Do something if the prediction is 1
        # elif prediction_index == 2:
        #     #Do something if the prediction is 2
        # #...

        events()
        refresh_win(display_frame, prediction)


if __name__ == '__main__':
    main()