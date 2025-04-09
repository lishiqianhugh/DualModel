import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import pymunk
import pymunk.pygame_util
import random
import math


class PourSimulator:
    def __init__(self, width=1200, height=800, seed=2023, name='foo'):
        random.seed(seed)
        self.width = width
        self.height = height
        self.name = name

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        self.space = pymunk.Space()
        self.space.iterations = 60
        self.space.gravity = (0.0, 900.0)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        object_color = pymunk.space_debug_draw_options.SpaceDebugColor(0, 0, 0, 0)
        self.draw_options.shape_outline_color = object_color
        self.draw_options.shape_dynamic_color = object_color
        
        self.objects = []
        self.cups = []
        self.cup_centers = []
        self.cup_sizes = []
        self.cup_segments = []
        self.motors = []

        self.step = 0
        self.max_step = 70000
        self.motor_activation_step = 1200
        self.rotate_rate = -0.004
        self.ground = self.add_ground()

    def add_ground(self):
        ground = pymunk.Body(body_type=pymunk.Body.STATIC)
        segment = pymunk.Segment(ground, (0, self.height-10), (self.width, self.height-10), 10)
        segment.elasticity = 0.1
        segment.friction = 0.0
        self.space.add(ground, segment)

    def add_object(self, x, y, size=5, mass=1, object_shape="circle"):
        body = pymunk.Body(mass, 1)  # Initialize with moment of 1, will update below
        body.position = x, y

        if object_shape == "circle":
            radius = size / 2
            inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
            shape = pymunk.Circle(body, radius, (0, 0))
        elif object_shape == "rectangle":
            side_length = size
            if random.random() > 0.5:
                inertia = pymunk.moment_for_box(mass*2, (side_length * 0.5, side_length * 2))
                shape = pymunk.Poly.create_box(body, (side_length * 0.5, side_length * 2))
            else:
                inertia = pymunk.moment_for_box(mass*2, (side_length * 2, side_length * 0.5))
                shape = pymunk.Poly.create_box(body, (side_length * 2, side_length * 0.5))
        elif object_shape == "square":
            side_length = size
            inertia = pymunk.moment_for_box(mass, (side_length, side_length))
        elif object_shape == "triangle":
            base_length = size
            height = size * 0.866
            vertices = [(- base_length / 2, - height / 3), (base_length / 2, - height / 3), (0, 2 * height / 3)]
            inertia = pymunk.moment_for_poly(mass, vertices)
            shape = pymunk.Poly(body, vertices)
            segments = [
                # pymunk.Segment(body, vertices[0], vertices[1], 5), 
                # pymunk.Segment(body, vertices[1], vertices[2], 5), 
                # pymunk.Segment(body, vertices[2], vertices[0], 5)  
            ]
        elif object_shape == "trapezoid":
            top_width = size * 1.0
            bottom_width = size * 1.5
            height = size * 1.0
            vertices = [(-height / 2, -bottom_width / 2),
                        (height / 2, -top_width / 2),
                        (height / 2, top_width / 2),
                        (-height / 2, bottom_width / 2)]

            inertia = pymunk.moment_for_poly(mass, vertices)
            shape = pymunk.Poly(body, vertices)
        else:
            raise ValueError(f"Unknown object_shape: {object_shape}")

        body.moment = inertia  # Update the moment of inertia now that we have it

        shape.elasticity = 0
        shape.friction = 0
        shape.color = [150 * random.random()] * 3 + [255]
        # change angle of bodies
        body.angle = random.random() * math.pi

        self.space.add(body, shape)
        self.objects.append(shape)

        if object_shape == 'triangle':
            for segment in segments:
                segment.elasticity = 0
                segment.friction = 0
                segment.color = shape.color
                self.space.add(segment)
                self.objects.append(segment)


    def add_cup(self, x, y, cup_width, cup_height, cup_thickness=4, cup_shape='regular'):
        cup_center = [x + cup_width / 2, y + cup_height / 2]
        self.cup_centers.append(cup_center)
        self.cup_sizes.append([cup_width, cup_height])
        
        cup_body = pymunk.Body() # body_type=pymunk.Body.STATIC)
        cup_body.position = cup_center
        self.space.add(cup_body)
        cup = []

        if cup_shape == 'regular':
            cup.append(self.create_segment((x, y + cup_height), (x + cup_width, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
            cup.append(self.create_segment((x, y), (x, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
            cup.append(self.create_segment((x + cup_width, y), (x + cup_width, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
        elif cup_shape == 'erlenmeyer':
            offset = -cup_width / 8
            cup.append(self.create_segment((x + offset, y + cup_height), (x + cup_width - offset, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
            cup.append(self.create_segment((x - offset, y), (x + offset, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
            cup.append(self.create_segment((x + offset + cup_width, y), (x + cup_width - offset, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
        elif cup_shape == 'ierlenmeyer':
            offset = cup_width / 8
            cup.append(self.create_segment((x + offset, y + cup_height), (x + cup_width - offset, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
            cup.append(self.create_segment((x - offset, y), (x + offset, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
            cup.append(self.create_segment((x + offset + cup_width, y), (x + cup_width - offset, y + cup_height), 
                                        cup_thickness, cup_body, cup_center))
        else:
            raise ValueError(f"Unknown cup_shape: {cup_shape}")
        self.cup_segments.append(cup)
        for s in cup:
            self.space.add(s)

        self.space.add(pymunk.PivotJoint(cup_body, self.space.static_body, cup_center))
        c = pymunk.SimpleMotor(cup_body, self.space.static_body, 0.0)
        self.space.add(c)
        self.motors.append(c)
        self.cups.append(cup_body)
    
    def create_segment(self, p1, p2, thickness, cup_body, cup_center):
        segment = pymunk.Segment(cup_body, (p1[0] - cup_center[0], p1[1] - cup_center[1]), 
                                 (p2[0] - cup_center[0], p2[1] - cup_center[1]), thickness)
        segment.mass = 10
        segment.elasticity = 0.0
        segment.friction = 0.0
        segment.color = (180, 180, 180, 255)
        return segment

    def add_noise(self, position_noise=0, angle_noise=0, cup_angle_noise=0, size_noise=0, velocity_noise=0):
        # sort objects by y position and select the highest 7 objects (small y means high position)
        # self.objects.sort(key=lambda x: x.body.position.y)
        # for object in self.objects[:7]:
        for object in self.objects:
            if angle_noise:
                object.body.angle += random.gauss(0, angle_noise)
            if position_noise:
                object.body.position += (random.gauss(0, position_noise), random.gauss(0, position_noise))
                # object.body.position += (random.gauss(-2.5, position_noise / 2), random.gauss(-5, position_noise))
                # object.body.position += (-abs(random.gauss(0, position_noise / 2)), -abs(random.gauss(0, position_noise)))
                # object.body.position += (position_noise, position_noise)
            if velocity_noise:
                # object.body.velocity += (-abs(random.gauss(0, velocity_noise)), -abs(random.gauss(0, velocity_noise)))
                object.body.velocity *= random.gauss(1, velocity_noise)
                # object.body.velocity += (velocity_noise, velocity_noise)
        if cup_angle_noise:    
            for cup in self.cups:
                cup.angle += random.gauss(0, cup_angle_noise)

    def simulate(self, save_init_dir=None, save_seq_dir=None, save_interval=100, 
                 ref_line=False, fps=120, tick=480, 
                 position_noise=0, angle_noise=0, cup_angle_noise=0, size_noise=0, velocity_noise=0,
                 noise_step=1, fast_mode=False):
        save = False
        init_info = None
        pouring_angle = None
        while self.step < self.max_step:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.step = self.max_step
            if (position_noise or angle_noise or cup_angle_noise or size_noise or velocity_noise) and self.step > self.motor_activation_step and self.step % noise_step == 0:
                self.add_noise(position_noise, angle_noise, cup_angle_noise, size_noise, velocity_noise)
            self.space.step(1/fps)
            self.screen.fill((255,255,255))
            self.space.debug_draw(self.draw_options)
            if ref_line:
                self.draw_angle_line()
                # if pouring_angle:
                #     self.draw_text(text=f'{round(pouring_angle)}', position=(self.width/2-70, self.height/2-50), size=50, color=(255, 0, 0))
            pygame.display.flip()
            self.clock.tick(tick)
            angle = self.detect_pour_out()
            if self.step > self.motor_activation_step and angle is not None and pouring_angle is None:
                print(f"Objects started pouring out at angle: {angle}")
                pouring_angle = angle
                if fast_mode:
                    return init_info, pouring_angle
                if ref_line and save and save_seq_dir != None:
                    self.draw_text(text=f'{round(pouring_angle)}', position=(self.width/2-70, self.height/2-50), size=50, color=(255, 0, 0))
                    for i in range(1, 30):
                        pygame.image.save(self.screen, f'{save_seq_dir}/{int((self.step-self.motor_activation_step)/save_interval)}.{i:02d}.png')
            if self.step < self.motor_activation_step:
                for cup in self.cups:
                    cup.angle = 0  # reset cup angle
            if self.step == self.motor_activation_step:
                # print("Heuristic angle: ", self.heuristc_angle())
                save = True
                init_info = {'space': self.space, 'objects': self.objects, 'cups': self.cups, 
                             'motors': self.motors, 'cup_centers': self.cup_centers, 'cup_sizes': self.cup_sizes, 
                             'cup_segments': self.cup_segments, 'motor_activation_step': self.motor_activation_step,
                             'rotate_rate': self.rotate_rate, 'max_step': self.max_step}
                for motor in self.motors:
                    motor.rate = self.rotate_rate
                if save_init_dir != None:
                    if not os.path.exists(save_init_dir):
                        os.makedirs(save_init_dir)
                    pygame.image.save(self.screen, f'{save_init_dir}/{self.name}.png')
            if save and save_seq_dir != None and self.step % save_interval == 0:
                if not os.path.exists(save_seq_dir):
                    os.makedirs(save_seq_dir)
                pygame.image.save(self.screen, f'{save_seq_dir}/{int((self.step-self.motor_activation_step)/save_interval)}.png')
            self.step += 1
        return init_info, pouring_angle

    def detect_pour_out(self):
        pouring_angle = None

        for object in self.objects:
            object_x, object_y = object.body.position

            for i, cup in enumerate(self.cups):
                cup_center_x, cup_center_y = self.cup_centers[i]
                cup_angle = cup.angle

                # get the initial top-left corner of the cup to calculate the offset
                topleft_x_offset = - self.cup_segments[i][1].a.x
                topleft_y_offset = - self.cup_segments[i][1].a.y

                # Calculate the rotated coordinates for the top-left corner of the cup
                rotated_x1 = cup_center_x - topleft_x_offset * math.cos(-cup_angle) - topleft_y_offset * math.sin(-cup_angle)
                rotated_y1 = cup_center_y + topleft_x_offset * math.sin(-cup_angle) - topleft_y_offset * math.cos(-cup_angle)
                # Check if the object is outside the rotated cup and moving downward
                if object_x < rotated_x1 and object_y < rotated_y1 and object.body.velocity.y > 0:
                    pouring_angle = -math.degrees(cup.angle)
                    return pouring_angle

        return pouring_angle  # Returns None if no objects are pouring out

    def draw_angle_line(self):
        for i, cup in enumerate(self.cups):
            cup_center_x, cup_center_y = self.cup_centers[i]
            cup_angle = cup.angle
            length = 300
            ori_p1 = (cup_center_x, cup_center_y -length)
            ori_p2 = (cup_center_x, cup_center_y + length)
            p1 = (cup_center_x - length * math.sin(-cup_angle), cup_center_y - length * math.cos(-cup_angle))
            p2 = (cup_center_x + length * math.sin(-cup_angle), cup_center_y + length * math.cos(-cup_angle))
            pygame.draw.lines(self.screen, (255, 0, 0), False, [ori_p1, ori_p2])
            pygame.draw.lines(self.screen, (255, 0, 0), False, [p1, p2])
            # draw line at the top left corner of the cup
            topleft_x_offset = - self.cup_segments[i][1].a.x
            topleft_y_offset = - self.cup_segments[i][1].a.y
            rotated_x1 = cup_center_x - topleft_x_offset * math.cos(-cup_angle) - topleft_y_offset * math.sin(-cup_angle)
            rotated_y1 = cup_center_y + topleft_x_offset * math.sin(-cup_angle) - topleft_y_offset * math.cos(-cup_angle)
            pygame.draw.lines(self.screen, (255, 0, 0), False, [(rotated_x1, rotated_y1-20), (rotated_x1, rotated_y1+20)])
    
    def draw_text(self, text, position, size=24, color=(0, 0, 0)):
        font = pygame.font.Font(None, size)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def heuristc_angle(self):
        # calculate the angle between the top-left corner of the cup (initial scene) and the top right object
        assert len(self.cups) == 1
        cup_center_x, cup_center_y = self.cup_centers[0]
        # print(f"Cup center: ({cup_center_x}, {cup_center_y})")
        cup_width, cup_height = self.cup_sizes[0]
        top_left_x = cup_center_x - cup_width / 2
        top_left_y = cup_center_y - cup_height / 2
        # print(f"Top left corner: ({top_left_x}, {top_left_y})")
        # find the top right object
        top_right_object_x = cup_center_x
        top_right_object_y = self.height
        for object in self.objects:
            _, object_y = object.body.position
            if object_y <= top_right_object_y:
                top_right_object_y = object_y
        
        # print(f"Top right object: ({top_right_object_x}, {top_right_object_y})")
        angle = math.degrees(math.atan2(top_right_object_y - top_left_y, top_right_object_x - top_left_x))

        return angle
