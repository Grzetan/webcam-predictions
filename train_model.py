import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pygame
from config import Config
config = Config()

def main(outputs,network, name):

    pygame.init()

    W, H = 640, 700
    WIN = pygame.display.set_mode((W,H))
    pygame.display.set_caption('Training model')
    CLOCK = pygame.time.Clock()
    FPS = 30

    run = True
    FONT = pygame.font.SysFont('comicsans', 50)
    status = 'waiting'

    train_data = pickle.load(open('dataset.pickle', 'rb'))
    np.random.shuffle(train_data)

    optimizer = optim.Adam(network.parameters(), lr=config.lr)

    torch.set_grad_enabled(True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size)

    def get_num_correct(preds, label):
        return preds.argmax(dim=1).eq(label).sum().item()

    def events():
        nonlocal run
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                sys.exit()

    def refresh_win(i, batch, epoch,epochs):
        WIN.fill(0)

        TEXT = FONT.render('Training model. Please wait', 1, (255,0,0))
        WIN.blit(TEXT, (W//2 - TEXT.get_width()//2, H//2-TEXT.get_height()//2))

        total = batch * epochs
        current = i + batch * epoch
        percent = current/total
        pygame.draw.rect(WIN, (255,255,255), (W//2 - 200, H-200, 400,30))
        pygame.draw.rect(WIN, (255,0,0), (W//2 - 200, H-200, percent*400,30))

        pygame.display.update()

    for epoch in range(config.epochs):
        total_loss = 0
        num_correct = 0

        for i,batch in enumerate(train_loader):
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss
            num_correct += get_num_correct(preds, labels)

            CLOCK.tick(FPS)
            refresh_win(i, len(train_loader), epoch,3)
            events

        print(f'Epoch {epoch}: total loss = {total_loss} {num_correct/len(train_data)}')

    torch.save(network, name + '.pth')
