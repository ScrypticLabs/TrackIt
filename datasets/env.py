import pygame
import pygame.camera
from pygame.locals import *
from pygame.color import *
import pymunk
from pymunk import Vec2d
import random
import json
from datetime import datetime
import os
import copy

def transform_pymunk_pygame(coordinates, window_dim):
    x,y = coordinates[0], coordinates[1]
    return (x, window_dim[1]-y)

class Environment():
    def __init__(self):
        pygame.init()

        self.RANGE_OF_BALLS = (1, 6)
        self.NUM_OF_BALLS = None
        self.VELOCITY_RANGE = (500, 600)

        self.window_dim = (1000, 1000)
        self.fullscreen = False
        self.screen = None          # will be set after load()
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.running = True         # window status
        self.simulate = True        # pause/resume physics engine

        # config file name
        self.file = None
        self.frame = 0

        self.space = pymunk.Space() # physics simulator 
        self.space.gravity = (0,0)

        self.loaded_config = False

        self.walls = []             # walls and balls must be built before start()
        self.balls = []

        self.RECORD = True
        self.check_directory = False
        
    def build_walls(self):
        corners = [(0,0), (self.window_dim[0], 0), (self.window_dim[0], self.window_dim[1]), (0, self.window_dim[1]), (0,0)]
        self.walls = []
        for i in range(len(corners)-1):
            self.walls.append(Wall(corners[i], corners[i+1], elasticity=1.0, friction=0))
        for wall in self.walls:
            self.space.add(wall.shape)

    def build_balls(self, balls=None):
        self.balls = []
        if balls:
            self.balls = balls
        else:
            if self.NUM_OF_BALLS:
                for _ in range(self.NUM_OF_BALLS):
                    self.balls.append(Ball(mass=50, position=(random.randint(0, self.window_dim[0]), random.randint(0, self.window_dim[1])), angular_position=0, velocity=(random.choice([-1,1])*random.randint(*self.VELOCITY_RANGE), random.choice([-1,1])*random.randint(*self.VELOCITY_RANGE)), angular_velocity=0, elasticity=1.0, friction=0.0, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))))
            else:
                for _ in range(random.randint(self.RANGE_OF_BALLS[0], self.RANGE_OF_BALLS[1])):
                    self.balls.append(Ball(mass=50, position=(random.randint(0, self.window_dim[0]), random.randint(0, self.window_dim[1])), angular_position=0, velocity=(random.choice([-1,1])*random.randint(*self.VELOCITY_RANGE), random.choice([-1,1])*random.randint(*self.VELOCITY_RANGE)), angular_velocity=0, elasticity=1.0, friction=0.0, color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))))
        for ball in self.balls:
            self.space.add(ball.shape.body, ball.shape)

    def start(self, n=None):
        if n:
            self.NUM_OF_BALLS = n
        if not self.loaded_config: self.load()
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    self.running = False
                elif event.type == KEYDOWN and event.key == K_SPACE:
                    self.simulate = not self.simulate
                elif event.type == KEYDOWN and event.key == K_s:
                    self.save()
                elif event.type== KEYDOWN and event.key == K_r:
                    self.RECORD = not self.RECORD
                    print(self.RECORD)
            if self.simulate:
                dt = 1.0/self.fps
                for _ in range(1):
                    self.space.step(dt)

            #-------------MAIN-LOOP----------------
            self.screen.fill((0,0,0))
            for wall in self.walls:
                wall.render(self.window_dim, self.screen)
            for ball in self.balls:
                ball.render(self.window_dim, self.screen)

            # record screen
            if self.RECORD:
                folder_name = self.file.split(".")[0]
                if not self.check_directory:
                    if not os.path.exists(os.getcwd()+"/videos/"+str(self.NUM_OF_BALLS)+'/'+folder_name):
                        os.makedirs(os.getcwd()+"/videos/"+str(self.NUM_OF_BALLS)+'/'+folder_name)
                    self.check_directory = True
                pygame.image.save(self.screen, "videos/%s/%s/%s_%04d.png" % (str(self.NUM_OF_BALLS), folder_name, folder_name, self.frame))                        
                self.frame += 1
                if self.frame >= 500:
                    self.RECORD = False
                    self.running = False
            # record screen
            
            pygame.display.flip()
            self.clock.tick(self.fps)
            pygame.display.set_caption("FPS: " + str(self.clock.get_fps()))
        pygame.quit()

    # always call save before start() because you want to save the initial environment state
    def save(self):
        config = {}
        config["window"] = {
            "width" : self.window_dim[0],
            "height": self.window_dim[1],
            "fullscreen" : self.fullscreen
        }
        config["balls_count"] = self.NUM_OF_BALLS
        config["fps"] = self.fps
        config["gravity"] = {
            "x" : self.space.gravity[0],
            "y" : self.space.gravity[1]
        }
        config["balls"] = []
        for ball in self.balls:
            config["balls"].append(ball.generate_config())
        
        # version 1 of environment
        v1 = copy.deepcopy(config)
        self.save_config(v1, version="1")
        
        # version 2 of environment
        v2 = copy.deepcopy(config)
        for ball_config in v2["balls"]:
            ball_config["velocity"]["x"] *= -1
        self.save_config(v2, version="2")

        # version 3 of environment
        v3 = copy.deepcopy(config)
        for ball_config in v3["balls"]:
            ball_config["velocity"]["y"] *= -1
        self.save_config(v3, version="3")

        # version 4 of environment
        v4 = copy.deepcopy(config)
        for ball_config in v4["balls"]:
            ball_config["velocity"]["x"] *= -1
            ball_config["velocity"]["y"] *= -1
        self.save_config(v4, version="4")
        
    def save_config(self, config, version=None):
        filename = None
        if version:
            filename = "config/"+str(self.NUM_OF_BALLS)+"/"+str(self.NUM_OF_BALLS)+"__"+version+datetime.now().strftime("__%d_%m_%Y__%H_%M_%S.json")
        else:
            filename = "config/"+str(self.NUM_OF_BALLS)+"/"+str(self.NUM_OF_BALLS)+"__"+datetime.now().strftime("%d_%m_%Y__%H_%M_%S.json")
        with open(filename, "w") as file:
            json.dump(config, file)

    def load(self, file=None):
        balls = None
        if file:
            with open(file, "r") as f:
                config = json.load(f)
                self.window_dim = (config["window"]["width"], config["window"]["height"])
                self.fullscreen = config["window"]["fullscreen"]
                self.fps = config["fps"]
                self.NUM_OF_BALLS = config["balls_count"]
                self.space.gravity = (config["gravity"]["x"], config["gravity"]["y"])
                balls = []
                for ball_config in config["balls"]:
                    color = (ball_config["color"]["r"], ball_config["color"]["g"], ball_config["color"]["b"])
                    fill = ball_config["color"]["fill"]
                    mass = ball_config["mass"]
                    position = (ball_config["position"]["x"], ball_config["position"]["y"])
                    angular_position = ball_config["angular_position"]
                    velocity = (ball_config["velocity"]["x"], ball_config["velocity"]["y"])
                    angular_velocity = ball_config["angular_velocity"]
                    elasticity = ball_config["elasticity"]
                    friction = ball_config["friction"]
                    show_spin = ball_config["show_spin"]
                    balls.append(Ball(mass=mass, position=position, velocity=velocity, angular_position=angular_position, angular_velocity=angular_velocity, elasticity=elasticity, friction=friction, color=color, fill=fill, show_spin=show_spin))
            self.file = file.split("/")[-1]
        else:
            self.file = "random"
        self.build_walls()
        self.build_balls(balls)
        if self.fullscreen:
            self.screen = pygame.display.set_mode(self.window_dim, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.window_dim)
        self.loaded_config = True
        
class Wall():
    def __init__(self, start_pos, end_pos, elasticity, friction):
        start, end = Vec2d(*start_pos), Vec2d(*end_pos)
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(body, start, end, radius=0.0)
        self.shape.friction = friction
        self.shape.elasticity = elasticity
        # self.shape.mass = mass
    
    def render(self, window_dim, surface, color=(0,0,0)):
        body = self.shape.body
        start = body.position + self.shape.a.rotated(body.angle)
        end = body.position + self.shape.b.rotated(body.angle)
        pygame.draw.lines(surface, color, False, [transform_pymunk_pygame((start.x, start.y), window_dim), transform_pymunk_pygame((end.x, end.y), window_dim)])

class Ball():
    def __init__(self, mass, position, angular_position, velocity, angular_velocity, elasticity, friction, color=(0,0,255), fill=True, show_spin=False, COLLTYPE_BALL=2):
        body = pymunk.Body(mass=mass, moment=100) 
        body.position = position
        body.angular_position = angular_position
        body.velocity = velocity
        body.angular_velocity = angular_velocity

        self.show_spin=show_spin
        self.fill = fill
        self.color = color
        
        self.shape = pymunk.Circle(body, radius=self.compute_radius_from_mass(mass), offset=(0,0))
        self.shape.elasticity = elasticity
        self.shape.friction = friction
        self.shape.collision_type = COLLTYPE_BALL
    
    def compute_radius_from_mass(self, mass):
        return mass

    def generate_config(self):
        config = {}
        config["color"] = {
            "r" : self.color[0],
            "g" : self.color[1],
            "b" : self.color[2],
            "fill" : self.fill
        }
        config["mass"] = self.shape.body.mass
        config["position"] = {
            "x" : self.shape.body.position[0],
            "y" : self.shape.body.position[1]
        }
        config["angular_position"] = self.shape.body.angular_position
        config["velocity"] = {
            "x" : self.shape.body.velocity[0],
            "y" : self.shape.body.velocity[1]
        }
        config["angular_velocity"] = self.shape.body.angular_velocity
        config["elasticity"] = self.shape.elasticity
        config["friction"] = self.shape.friction
        config["show_spin"] = self.show_spin
        return config

    def render(self, window_dim, surface):
        r = int(self.shape.radius)
        v = self.shape.body.position
        rot = self.shape.body.rotation_vector
        p = tuple([int(x) for x in transform_pymunk_pygame(v, window_dim)])
        p2 = Vec2d(rot.x, -rot.y) * r * 0.9
        pygame.draw.circle(surface, self.color, p, r, 0 if self.fill else 2)
        if self.show_spin:
            pygame.draw.line(surface, (255,255,255), p, p+p2)

class Agent():
    def __init__(self):
        pass

if __name__ == '__main__':
    env = Environment()
    # env.load(file="config/31_10_2019__14_00_09_experimental.json")
    # env.load(file="config/31_10_2019__14_00_09.json")
    # env.start(n=5)
    env.load(file="config/1/1__1__02_01_2020__16_53_57.json")
    env.start()
    

# mencoder videos/31_10_2019__14_00_09/*.png -mf w=1000:h=1000:fps=60:type=png -ovc lavc -lavcopts vcodec=msmpeg4v2:vbitrate=16000:keyint=15:mbd=2:trell -oac copy -o output.avi

