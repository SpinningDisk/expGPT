import pygame
import random
import numpy as np


def drawGrid(blocksize, width, screen):
    blocksize = 2*blocksize
    for x in range(0, 2*blocksize**2, blocksize):
        for y in range(0, 2*blocksize**2, blocksize):
            rect = pygame.Rect(x, y, blocksize, blocksize)
            pygame.draw.rect(screen, (0, 0, 0), rect, width)

class Apple():
    def __init__(self, initial_amount: int, constant_amount: bool, color: tuple|list, board_size: tuple|list):
        self.pos = []
        for i in range(initial_amount):
            self.pos.append([random.randint(0, board_size[0]), random.randint(0, board_size[1])])    
            while 2 in set([self.pos.count(n) for n in self.pos]):
                self.pos.pop()
                self.pos.append([random.randint(0, board_size[0]), random.randint(0, board_size[1])])
        self.color = color
        self.board_size = board_size
        self.constant_amount = constant_amount
        if not constant_amount:
            # np is called latter on to determine weighted number of new apples (eat_by_index)
            from numpy.random import choice
            self.choice = choice
            print("imported np")
    def eat_by_index(self, index):
        if self.constant_amount:
            eaten = self.pos.pop(index)
            self.pos.append([random.randint(0, self.board_size[0]-1), random.randint(0, self.board_size[1]-1)])
            while self.pos[len(self.pos)-1] == eaten or 2 in set([self.pos.count(n) for n in self.pos]):
                self.pos.pop()
                self.pos.append([random.randint(0, self.board_size[0]-1), random.randint(0, self.board_size[1]-1)])
        else:
            eaten = self.pos.pop(index)
            match(len(self.pos)):
                case 1:
                    self.pos.append([random.randint(0, self.board_size[0]-1), random.randint(0, self.board_size[1]-1)])
                    while self.pos[len(self.pos)-1] == eaten or 2 in set([self.pos.count(n) for n in self.pos]):
                        self.pos.pop()
                        self.pos.append([random.randint(0, self.board_size[0]-1), random.randint(0, self.board_size[1]-1)])
                case _:
                    amount = self.choice([x for x in range(3)], 1, [(1/x)*10/len(self.pos) for x in range(1, 3)])
                    for i in range(int(amount)):
                        self.pos.append([random.randint(0, self.board_size[0]-1), random.randint(0, self.board_size[1]-1)])
                        while self.pos[len(self.pos)-1] == eaten or 2 in set([self.pos.count(n) for n in self.pos]):
                            elm = self.pos.pop()
                            self.pos.append([random.randint(0, self.board_size[0]-1), random.randint(0, self.board_size[1]-1)])


class Snake():
    def __init__(self, board_size: list|tuple, initial_length: int = 4, start_pos: list = [6, 4], dir: list|tuple=[0, 0], token_size: int|list|tuple = 3):
        self.pos = [start_pos]
        for i in range(1, initial_length):
            self.pos.append([start_pos[0]-i, start_pos[1]])
        if self.pos[initial_length-1][0] <= 0:
            print("\033[1;32mWarning!SnakeEnvBack: body partially out of bounds! this might be bad\033[0m")
        self.score = initial_length
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.goal = Apple(initial_amount=5, constant_amount=False, color=(255-self.color[0], 255-self.color[1], 255-self.color[2]), board_size=board_size)
        self.dir = dir
        self.board_size = board_size
        if type(token_size) == int:
            self.token_size = (token_size, )
        elif type(token_size) == list:
            self.token_size = tuple(token_size)
    def move(self):
        if [self.pos[0][0]+self.dir[0], self.pos[0][1]+self.dir[1]] == self.pos[1]:
            return
        match sum(self.dir):
            case 0:
                pass
            case _:
                for i in reversed(range(1, len(self.pos))):
                    self.pos[i] = self.pos[i-1].copy()
                self.pos[0] = [self.pos[0][0]+self.dir[0], self.pos[0][1]+self.dir[1]].copy()
    def eat(self):
        if self.pos[0] in self.goal.pos:
            self.goal.eat_by_index(self.goal.pos.index(self.pos[0]))
            if self.goal.pos[len(self.goal.pos)-1] in self.pos:
                self.goal.eat_by_index(len(self.goal.pos)-1)
            dx = self.pos[-2][0] - self.pos[-1][0]
            dy = self.pos[-2][1] - self.pos[-1][1]
            self.pos.append([self.pos[-1][0]+dx, self.pos[-1][1]+dy])
            self.score += 1
            return 1
    def check_collision(self):
        walls = []
        for x in range(self.board_size[0]):
            walls.append([-1, x])
            walls.append([self.board_size[0], x])
            walls.append([x, -1])
            walls.append([x, self.board_size[0]])
        if self.pos[0] in walls:
            self.score -= 10
            return 1
        elif 2 in set([self.pos.count(n) for n in self.pos]):
            self.score -= 5
            return 2
    def state(self):
        board_elms = (self.board_size[0]+2)**2
        for i in self.token_size:
            board_elms *= i
        board = np.zeros(board_elms).reshape((self.board_size[0]+2, self.board_size[1]+2)+self.token_size)
        walls = []
        for x in range(self.board_size[0]+1):
            walls.append([0, x])
            walls.append([self.board_size[0]+1, x])
            walls.append([x, 0])
            walls.append([x, self.board_size[0]+1])
        walls.append([self.board_size[0]+1, self.board_size[0]+1])
        for i in walls:
            board[i[1], i[0], ...] = 1
        for i in self.pos:
            board[i[1]+1, i[0]+1, ...] = 2
        for i in self.goal.pos:
            board[i[1]+1, i[0]+1, ...] = 3
        board[self.pos[0][1]+1, self.pos[0][0]+1, ...] = 4
        return board
    def draw(self, screen, blocksize):
        def grid_to_rect(coords, color, blocksize, screen):
            rect = pygame.Rect(32 * coords[0], 32 * coords[1], 2 * blocksize, 2 * blocksize)
            pygame.draw.rect(screen, color, rect)
            return
        grid_to_rect(self.pos[0], (200, 10, 15), blocksize, screen)
        for piece in range(1, len(self.pos)):
            draw_pos = [self.pos[piece], -self.pos[piece][1]].copy()
            grid_to_rect(self.pos[piece], self.color, blocksize, screen)
        for food in self.goal.pos:
            grid_to_rect(food, self.goal.color, blocksize, screen)


if __name__=="__main__":
    grid_size = (16, 16)

    snek = Snake(initial_length=7, dir=[0, 0], board_size=grid_size)
    pygame.init()

    screen = pygame.display.set_mode((2*grid_size[0]**2, 2*grid_size[0]**2))
    
    running = True
    while running:
        print(len(snek.pos))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 200, 30))
        snek.draw(screen, grid_size[0])
        if pygame.key.get_pressed()[pygame.K_w]:
            snek.dir = [0, -1]
        elif pygame.key.get_pressed()[pygame.K_s]:
            snek.dir = [0, 1]
        elif pygame.key.get_pressed()[pygame.K_a]:
            snek.dir = [-1, 0]
        elif pygame.key.get_pressed()[pygame.K_d]:
            snek.dir = [1, 0]
        elif pygame.key.get_pressed()[pygame.K_BACKSPACE]:
            snek.state()
        snek.eat()
        snek.move()
        if snek.check_collision():
            snek = Snake(initial_length=4, dir=[0, 0], board_size=grid_size)
        snek.state()
        drawGrid(grid_size[0], 3, screen)
        pygame.display.update()
        pygame.time.Clock().tick(5)
    pygame.quit()
