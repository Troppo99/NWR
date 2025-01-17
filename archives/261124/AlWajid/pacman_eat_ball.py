import sys
import os
import pygame
import math

# Inisialisasi Pygame
pygame.init()

# Definisikan ukuran jendela
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pac-Man Sederhana")

# Definisikan warna
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Mengatur clock untuk kontrol FPS
clock = pygame.time.Clock()
FPS = 60

# Mengatur path untuk assets
ASSETS_PATH = os.path.join("main", "Al-Wajid", "assets")


# Memuat gambar Pac-Man dan bola
def load_image(name, scale=None):
    path = os.path.join(ASSETS_PATH, name)
    try:
        image = pygame.image.load(path).convert_alpha()
    except pygame.error as e:
        print(f"Unable to load image {path}: {e}")
        sys.exit()
    if scale:
        image = pygame.transform.scale(image, scale)
    return image


# Memuat dan menskalakan gambar
PACMAN_IMAGES = [
    load_image("pacman1.png", scale=(60, 60)),
    load_image("pacman2.png", scale=(60, 60)),
    load_image("pacman3.png", scale=(60, 60)),
    load_image("pacman4.png", scale=(60, 60)),
]
BOLA_IMAGE = load_image("bola.png", scale=(30, 30))


class PacMan:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.direction = "RIGHT"  # Arah default
        self.radius = 30  # Radius Pac-Man
        self.images = PACMAN_IMAGES
        self.current_image = 0
        self.animation_speed = 0.1  # Kecepatan animasi
        self.animation_timer = 0

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_d]:
            self.direction = "RIGHT"
        else:
            self.direction = "STOP"

    def move(self):
        if self.direction == "RIGHT":
            self.x += self.speed
            # Membatasi gerakan agar tidak keluar layar
            if self.x > SCREEN_WIDTH - self.radius:
                self.x = SCREEN_WIDTH - self.radius

    def update_animation(self):
        if self.direction != "STOP":
            self.animation_timer += self.animation_speed
            if self.animation_timer >= len(self.images):
                self.animation_timer = 0
            self.current_image = int(self.animation_timer)
        else:
            self.current_image = 0  # Gambar mulut tertutup

    def draw(self, screen):
        image = self.images[self.current_image]
        rect = image.get_rect(center=(self.x, self.y))
        screen.blit(image, rect)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 15  # Radius bola
        self.eaten = False

    def draw(self, screen):
        if not self.eaten:
            rect = BOLA_IMAGE.get_rect(center=(self.x, self.y))
            screen.blit(BOLA_IMAGE, rect)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


def check_collision(pacman, ball):
    distance = math.hypot(pacman.x - ball.x, pacman.y - ball.y)
    return distance < (pacman.radius + ball.radius)


def main():
    # Membuat objek Pac-Man di posisi awal
    pacman = PacMan(x=100, y=SCREEN_HEIGHT // 2, speed=5)

    # Membuat tiga bola dalam garis lurus ke kanan
    balls = [
        Ball(x=300, y=SCREEN_HEIGHT // 2),
        Ball(x=450, y=SCREEN_HEIGHT // 2),
        Ball(x=600, y=SCREEN_HEIGHT // 2),
    ]

    # Menambahkan skor
    score = 0
    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        clock.tick(FPS)  # Kontrol FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Tangani input
        pacman.handle_keys()

        # Gerakkan Pac-Man
        pacman.move()

        # Update animasi Pac-Man
        pacman.update_animation()

        # Deteksi tabrakan dengan bola
        for ball in balls:
            if not ball.eaten and check_collision(pacman, ball):
                ball.eaten = True
                score += 10
                print("Bola dimakan!")

        # Cek kondisi menang
        if all(ball.eaten for ball in balls):
            print(f"Selamat! Anda telah memakan semua bola dengan skor {score}!")
            running = False

        # Gambar latar belakang
        screen.fill(BLACK)

        # Gambar bola
        for ball in balls:
            ball.draw(screen)

        # Gambar Pac-Man
        pacman.draw(screen)

        # Gambar skor
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        # Perbarui tampilan
        pygame.display.flip()

    # Keluar dari Pygame
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
