import pygame
import random
import sys

# 初始化pygame
pygame.init()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

# 游戏配置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 20
CELL_NUMBER_X = WINDOW_WIDTH // CELL_SIZE
CELL_NUMBER_Y = WINDOW_HEIGHT // CELL_SIZE

# 方向定义
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        # 蛇的初始位置（屏幕中央）
        self.body = [(CELL_NUMBER_X // 2, CELL_NUMBER_Y // 2)]
        self.direction = RIGHT
        self.new_block = False

    def update(self):
        """更新蛇的位置"""
        head_x, head_y = self.body[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        if self.new_block:
            self.body.insert(0, new_head)
            self.new_block = False
        else:
            self.body.insert(0, new_head)
            self.body.pop()

    def draw(self, surface):
        """绘制蛇"""
        for i, block in enumerate(self.body):
            x = block[0] * CELL_SIZE
            y = block[1] * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

            # 蛇头用不同颜色
            if i == 0:
                pygame.draw.rect(surface, GREEN, rect)
            else:
                pygame.draw.rect(surface, BLUE, rect)
            pygame.draw.rect(surface, BLACK, rect, 1)  # 边框

    def check_collision(self):
        """检查碰撞"""
        head = self.body[0]

        # 检查撞墙
        if (head[0] < 0 or head[0] >= CELL_NUMBER_X or
            head[1] < 0 or head[1] >= CELL_NUMBER_Y):
            return True

        # 检查撞自己
        if head in self.body[1:]:
            return True

        return False

    def add_block(self):
        """增加身体长度"""
        self.new_block = True

class Food:
    def __init__(self):
        self.generate()

    def generate(self):
        """生成食物"""
        self.x = random.randint(0, CELL_NUMBER_X - 1)
        self.y = random.randint(0, CELL_NUMBER_Y - 1)

    def draw(self, surface):
        """绘制食物"""
        rect = pygame.Rect(self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, RED, rect)
        pygame.draw.rect(surface, BLACK, rect, 1)

class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.game_over = False
        self.font = pygame.font.Font(None, 36)

    def update(self):
        """更新游戏状态"""
        if not self.game_over:
            self.snake.update()

            # 检查吃到食物
            if self.snake.body[0] == (self.food.x, self.food.y):
                self.snake.add_block()
                self.food.generate()
                self.score += 10

                # 确保食物不生成在蛇身上
                while (self.food.x, self.food.y) in self.snake.body:
                    self.food.generate()

            # 检查游戏结束
            if self.snake.check_collision():
                self.game_over = True

    def draw(self, surface):
        """绘制游戏"""
        surface.fill(WHITE)

        if not self.game_over:
            self.snake.draw(surface)
            self.food.draw(surface)
        else:
            # 显示游戏结束画面
            game_over_text = self.font.render("游戏结束!", True, RED)
            score_text = self.font.render(f"最终得分: {self.score}", True, BLACK)
            restart_text = self.font.render("按空格键重新开始", True, BLACK)

            # 文本居中显示
            surface.blit(game_over_text,
                        (WINDOW_WIDTH//2 - game_over_text.get_width()//2,
                         WINDOW_HEIGHT//2 - 60))
            surface.blit(score_text,
                        (WINDOW_WIDTH//2 - score_text.get_width()//2,
                         WINDOW_HEIGHT//2 - 20))
            surface.blit(restart_text,
                        (WINDOW_WIDTH//2 - restart_text.get_width()//2,
                         WINDOW_HEIGHT//2 + 20))

        # 显示分数
        score_text = self.font.render(f"分数: {self.score}", True, BLACK)
        surface.blit(score_text, (10, 10))

    def handle_keypress(self, key):
        """处理按键"""
        if self.game_over:
            if key == pygame.K_SPACE:
                self.restart()
        else:
            # 控制蛇的方向
            if key == pygame.K_UP and self.snake.direction != DOWN:
                self.snake.direction = UP
            elif key == pygame.K_DOWN and self.snake.direction != UP:
                self.snake.direction = DOWN
            elif key == pygame.K_LEFT and self.snake.direction != RIGHT:
                self.snake.direction = LEFT
            elif key == pygame.K_RIGHT and self.snake.direction != LEFT:
                self.snake.direction = RIGHT

    def restart(self):
        """重启游戏"""
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.game_over = False

def main():
    # 创建游戏窗口
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("贪吃蛇游戏")
    clock = pygame.time.Clock()

    # 创建游戏实例
    game = Game()

    print("贪吃蛇游戏已启动!")
    print("控制说明:")
    print("- 使用方向键控制蛇的移动")
    print("- 吃红色食物增加分数和长度")
    print("- 避免撞墙和撞到自己")
    print("- 游戏结束后按空格键重新开始")
    print("- 按ESC键退出游戏")

    running = True
    while running:
        # 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    game.handle_keypress(event.key)

        # 更新游戏状态
        game.update()

        # 绘制游戏
        game.draw(screen)

        # 更新显示
        pygame.display.flip()
        clock.tick(10)  # 游戏速度 (FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()