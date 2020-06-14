import argparse
from collect_frames import main as CollectFramesModule
from train_model import main as TrainModel
from model import Network
from predict import main as Predict
import pygame
import sys
import os

# parser = argparse.ArgumentParser(description='Parser')

# parser.add_argument('outputs', type=int, help='Number of outputs')
# parser.add_argument('--num_imgs', type=int, help='Cos')
# parser.add_argument('--wait_time', type=int, help='Time that program will wait after pressing ENTER to collect frames (In miliseconds).')
# parser.add_argument('--batch_size', type=int, help='Batch size')
# parser.add_argument('--epochs', type=int, help='Number of epochs')

# args = parser.parse_args()

# outputs = args.outputs
# num_imgs = args.num_imgs if args.num_imgs is not None else 500
# wait_time = args.wait_time if args.wait_time is not None else 2000
# batch_size = args.batch_size if args.batch_size is not None else 5
# epochs = args.epochs if args.epochs is not None else 5

def main():
    pygame.init()
    W, H = 640, 800
    WIN = pygame.display.set_mode((W,H))
    CLOCK = pygame.time.Clock()
    FPS = 30
    run2 = True
    FONT = pygame.font.SysFont('comicsans', 35)
    BTN_FONT = pygame.font.SysFont('comicsans', 40)
    outputs = None
    num_imgs = None
    model_name = None

    def get_models():
        dirr = os.listdir('./')
        models = ['Empty', 'Empty','Empty', 'Empty', 'Empty']
        i = 0
        for d in dirr:
            if '.pth' in d:
                models[i] = d
                i+=1
        return models

    models = get_models()

    def train_new_model():
        run = True
        COLOR_INACTIVE = pygame.Color('lightskyblue3')
        COLOR_ACTIVE = pygame.Color('dodgerblue2')

        class InputBox:

            def __init__(self, x, y, w, h, text=''):
                self.rect = pygame.Rect(x, y, w, h)
                self.color = COLOR_INACTIVE
                self.text = text
                self.txt_surface = FONT.render(text, True, self.color)
                self.active = False

            def handle_event(self, event):
                nonlocal run
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # If the user clicked on the input_box rect.
                    if self.rect.collidepoint(event.pos):
                        # Toggle the active variable.
                        self.active = not self.active
                    else:
                        self.active = False
                    # Change the current color of the input box.
                    self.color = COLOR_ACTIVE if self.active else COLOR_INACTIVE
                if event.type == pygame.KEYDOWN:
                    if self.active:
                        # if event.key == pygame.K_RETURN:
                        #     name = self.text
                        #     run = False

                        if event.key == pygame.K_BACKSPACE:
                            self.text = self.text[:-1]
                        else:
                            self.text += event.unicode
                        # Re-render the text.
                        self.txt_surface = FONT.render(self.text, True, self.color)

            def update(self):
                # Resize the box if the text is too long.
                width = max(200, self.txt_surface.get_width()+10)
                self.rect.w = width

            def draw(self, screen):
                # Blit the text.
                screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
                # Blit the rect.
                pygame.draw.rect(screen, self.color, self.rect, 2)

        name = InputBox(W//2 - 100, 150,800,40, 'network')
        outputs_box = InputBox(W//2 - 100,300,800,40)
        imgs_box = InputBox(W//2-100, 450,800,40, '500')
        error_str = ''

        def collected():
            nonlocal error_str
            nonlocal run
            nonlocal model_name, outputs, num_imgs
            num_img_str = imgs_box.text
            name_str = name.text
            outputs_str = outputs_box.text

            def RepresentsInt(s):
                try: 
                    int(s)
                    return True
                except ValueError:
                    return False
                    
            models = get_models()

            if name_str == '' or num_img_str == '' or outputs_str == '':
                error_str = "Fill in all input boxes"
            elif ' ' in name_str:
                error_str = 'Name can\'t contain spaces'
            elif not RepresentsInt(outputs_str) or not RepresentsInt(num_img_str):
                error_str = 'Fill in gaps with correct type'
            elif name_str+'.pth' in models:
                error_str = "This name is already used"
            else:
                error_str = ''
                outputs = int(outputs_str)
                num_imgs = int(num_img_str)
                model_name = name_str
                run = False

        def go_back():
            import main
            main.main()

        text = FONT.render('Name your model', 1, (255,0,0))
        text2 = FONT.render('num outputs', 1, (255,0,0))
        text3 = FONT.render('num imgs', 1, (255,0,0))

        while run:
            CLOCK.tick(FPS)
            pygame.display.update()
            WIN.fill(0)
            create_button('<--', (128,128,128), (200,200,200), 30,30,50,50, action=go_back)
            WIN.blit(text, (W//2 - text.get_width()//2, 100))
            WIN.blit(text2,(W//2-text2.get_width()//2, 250))
            WIN.blit(text3,(W//2-text3.get_width()//2, 400))
            if error_str != '':
                text4 = FONT.render(error_str, 1, (0,0,255))
                WIN.blit(text4, (W//2-text4.get_width()//2, 50))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    sys.exit()
                name.handle_event(event)
                outputs_box.handle_event(event)
                imgs_box.handle_event(event)

            outputs_box.update()
            outputs_box.draw(WIN)
            name.update()
            name.draw(WIN)            
            imgs_box.update()
            imgs_box.draw(WIN)

            create_button('Continue', (0,128,0), (0,255,0), W//2-200, H-250, 400,80, action=collected)

        network = Network(outputs)
        CollectFramesModule(outputs, num_imgs)
        TrainModel(outputs, network, model_name)
        models = get_models()
        index = models.index(model_name + '.pth')
        Predict(model_name + '.pth',index)

    #models functions

    def predict_from_model_1():
        if models[0] == 'Empty':
            train_new_model()
        else:
            Predict(models[0],0)

    def predict_from_model_2():
        if models[1] == 'Empty':
            train_new_model()
        else:
            Predict(models[1],1)

    def predict_from_model_3():
        if models[2] == 'Empty':
            train_new_model()
        else:
            Predict(models[2],2)

    def predict_from_model_4():
        if models[3] == 'Empty':
            train_new_model()
        else:
            Predict(models[3],3)

    def predict_from_model_5():
        if models[4] == 'Empty':
            train_new_model()
        else:
            Predict(models[4],4)

    def delete_model_1():
        os.remove(models[0])
        main()

    def delete_model_2():
        os.remove(models[1])
        main()

    def delete_model_3():
        os.remove(models[2])
        main()

    def delete_model_4():
        os.remove(models[3])
        main()

    def delete_model_5():
        os.remove(models[4])
        main()

    def create_button(text, color, highlight_color, x, y, width, height, action=None):
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

    def events():
        global run
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                sys.exit()

    predict_from_models = [predict_from_model_1,predict_from_model_2,predict_from_model_3, predict_from_model_4,predict_from_model_5]
    delete_models = [delete_model_1, delete_model_2, delete_model_3, delete_model_4, delete_model_5]

    def refresh_win():
        WIN.fill(0)
        video = cv2.VideoCapture(0)
        status = 'ready'
        if video is None or not video.isOpened():
            status = 'no webcam'
        if status == 'no webcam':
            text = FONT.render('Connect webcam to your device first',1,(255,0,0))
            WIN.blit(text, (W//2 - text.get_width()//2, H//2 - text.get_height()//2))
        elif status == 'ready':
            video.release()
            text = FONT.render('Choose from which model you want to predict',1,(255,0,0))
            WIN.blit(text, (W//2 - text.get_width()//2, 100))

            for i, model in enumerate(models):
                create_button(model, (0,128,0), (0,255,0), W//2-150, 300+i*80, 300,30,action=predict_from_models[i])
                if model != 'Empty':
                    create_button('X', (128,0,0), (255,0,0), W//2-150 + 300 + 30, 300+i*80,30,30,action=delete_models[i])

        pygame.display.update()

    while run2:
        CLOCK.tick(FPS)
        events()
        refresh_win()

main()

# def refresh_win():
#     WIN.fill(0)

#     text = FONT.render('Welcome', 1,(255,255,255))
#     WIN.blit(text, (W//2 - text.get_width()//2, 100))
#     create_button('Train new model',(0,0,128), (0,0,255), W//2 - 200, 300, 400,50, action=train_new_model)
#     create_button('Predict from an existing one',(0,0,128), (0,0,255), W//2 - 200, 400, 400,50, action=predict_from_existing_model)

#     pygame.display.update()
