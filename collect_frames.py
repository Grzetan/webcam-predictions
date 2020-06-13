import cv2 as cv2
import pygame as pygame
import pickle
import sys
import numpy as np
from config import Config
config = Config()

def main(outputs, num_imgs):
    pygame.init()
    W, H = 640, 800
    WIN = pygame.display.set_mode((W,H))
    pygame.display.set_caption('Collect data to train a model')
    CLOCK = pygame.time.Clock()
    FPS = 30
    video = cv2.VideoCapture(0)
    run = True
    num = 0
    saved_frames = 0
    train_size = num_imgs
    collected = []
    dataset = []
    status = 'waiting'
    FONT = pygame.font.SysFont('comicsans', 30)
    dataset_str = open('dataset.pickle', 'wb')

    def events():
        nonlocal run

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pickle_in_frames.close()
                pickle_in_labels.close()
                run = False

    def keys():
        nonlocal status, num
        keys = pygame.key.get_pressed()

        if keys[pygame.K_RETURN]:
            if status is not 'collecting':
                pygame.time.wait(config.wait_time)
                status = 'collecting'
                num += 1

        if keys[pygame.K_s]:
            pickle.dump(dataset, dataset_str)
            dataset_str.close()
            sys.exit()

    def refresh_win(frame):
        WIN.fill(0)

        #display live webcam

        WIN.blit(frame, (0,0))

        #display status

        if status == 'waiting':
            text = FONT.render('Waiting', 1, (255,255,255))
            WIN.blit(text, (W//2 - text.get_width()//2, H - 100))
        elif status == 'collecting':
            text = FONT.render(f'Collecting samples of number {num}', 1, (255,255,255))
            WIN.blit(text, (W//2 - text.get_width()//2, H-200))
            pygame.draw.rect(WIN, (255,255,255), (W//2 - 200, H - 100, 400, 30))
            precent = saved_frames/train_size * 100 // 1
            pygame.draw.rect(WIN, (255,0,0), (W//2 - 200, H - 100, precent/100 * 400, 30))

        #display collected numbers

        string = 'Collected: '
        for number in collected:
            string = string + f'{number}, '
        text = FONT.render(string, 1, (255,255,255))
        WIN.blit(text, (10, H-300))

        #hint

        TEXT = FONT.render('To start collecting frames press ENTER.',1,(255,255,255))
        WIN.blit(TEXT, (20,550))

        pygame.display.update()    

    while run:
        CLOCK.tick(FPS)

        check, frame = video.read()

        #make frame displayable
        display_frame = pygame.surfarray.make_surface(frame)
        display_frame = pygame.transform.rotate(display_frame, -90)

        if status == 'collecting':
            
            if saved_frames < train_size and not num in collected:

                #make frame less heavy

                train_frame = cv2.resize(frame, (160, 120))
                train_frame = cv2.cvtColor(train_frame, cv2.COLOR_BGR2GRAY)
                train_frame = np.array(train_frame)
                train_frame.resize(1,120,160)

                #save

                dataset.append((train_frame, num-1))

                saved_frames += 1
                print(f'Saved {saved_frames}/{train_size} of number: {num}')

            else:
                if num >= outputs:
                    break
                saved_frames = 0
                collected.append(num)
                status = 'waiting'

        events()
        keys()
        refresh_win(display_frame)

    pickle.dump(dataset, dataset_str)
    dataset_str.close()