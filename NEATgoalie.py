import turtle
import random
import neat
import os
import pickle

class Goalkeeper(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape('circle')
        self.color("blue")
        self.penup()
        self.goto(-200, 0)
        self.speed = 0

    def move_down(self):
        self.speed = -5

    def move_up(self):
        self.speed = 5

    def stop(self):
        self.speed = 0

    def move(self):
        self.sety(self.ycor() + self.speed)
        if self.ycor() < -75:
            self.goto(-200,-75)
        elif self.ycor() > 75:
            self.goto(-200,75)

    def collision(self, ball):
        if self.distance(ball) < 20:
            ball.goto(0,0)
            self.goto(-200,0)
            return True
        else:
            return False


class Ball(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape('circle')
        self.color('black')
        self.penup()
        self.goto(0, 0)
        self.dx = -10
        self.dy = random.randint(-5,5)
        self.x = 0

    def move(self):
        self.y = self.x % 50
        if self.y == 0:
            self.dy = random.randint(-5,5)
        self.setx(self.xcor() + self.dx)
        self.sety(self.ycor() + self.dy)
        if self.ycor() < -75 or self.ycor() > 75:
            self.dy *= -1
        self.x += 1

win = turtle.Screen()
win.bgcolor("green3")
win.title("training")
line = turtle.Turtle()
line.color("white")
line.pensize(5)
line.hideturtle()
line.penup()
line.goto(-200,-75)
line.pendown()
line.goto(-200,75)
line.penup()

# Define the fitness function
def eval_genomes(genomes, config):
    global win
    ball = Ball()
    goalkeeper = Goalkeeper()
    for i, (genome_id1, genome1) in enumerate(genomes):
        print("genome: ",i+1)
        genome1.fitness = 5
        goalkeeper.goto(-200,0)
        ball.goto(0,0)
        for x in range(100):
            net = neat.nn.FeedForwardNetwork.create(genome1, config)
            ball.move()
            output = net.activate((ball.ycor(), goalkeeper.ycor()))
            descision = output.index(max(output))
            if descision > 1:
                goalkeeper.move_up()    
                goalkeeper.move()
            elif descision < 1:
                goalkeeper.move_down()
                goalkeeper.move()
            else:
                goalkeeper.stop()
                goalkeeper.move()
            if goalkeeper.collision(ball):
                genome1.fitness += 1
                goalkeeper.goto(-200,0)
                ball.goto(0,0)
            if ball.xcor() <= -200:
                genome1.fitness -= 1
                goalkeeper.goto(-200,0)
                ball.goto(0,0)
        print(genome1.fitness)
    ball.hideturtle()
    goalkeeper.hideturtle()

# Define the NEAT function
def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-48")
    
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    # Run the NEAT algorithm for up to 50 generations
    winner = p.run(eval_genomes, 1)
    print(winner)

    # Print the winning genome
    print('\nBest genome:\n{!s}'.format(winner))
    with open("best.dump","wb") as f:
        pickle.dump(winner, f)

def test_ai(config):
    with open("best.dump", "wb") as f:
        winner = pickle.load(f)

# Run the NEAT function
if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    # Load the NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    run_neat(config)