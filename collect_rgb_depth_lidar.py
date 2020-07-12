#!/usr/bin/env python
import glob
import os
import sys
from PIL import Image as PImage
import numpy as np
import io
import datetime

# to visualize ply 3d point cloud
# https://stackoverflow.com/questions/50965673/python-display-3d-point-cloud

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import random
try:
    import queue
except ImportError:
    import Queue as queue

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

# TIME INTERVAL BETWEEN FRAMES
TIME_INTER = 1

# STOP AFTER HOW MANY MILLISECONDS
STOP_AFTER = 3600 * 10 * TIME_INTER


# define parameters
OTHER_VEH_NUM = 0
OTHER_PED_NUM = 0

# semantic segmentation sensor parameters
SENSOR_TICK = 0.0
SAVE_PATH = "./images/"
NAME_WITH_TIME = "_".join("_".join(str(datetime.datetime.now()).split(" ")).split(":"))[:16]
SENSOR_TYPE_1 = "CameraSemSeg"
SENSOR_TYPE_1_CARLA = "sensor.camera.semantic_segmentation"

# format of all image sensor data
WINDOW_HEIGHT = 375
WINDOW_WIDTH = 1242
BUFFER_SIZE = 100
FOV = 120

# lidar sensor parameters
SENSOR_TYPE_2 = 'Lidar'
SENSOR_TYPE_2_CARLA = "sensor.lidar.ray_cast"
CHA_NUM = 64
RANGE = 50
PTS_PER_SEC = 100000
ROT_FREQ = 10
UPPER_FOV = 10
LOWER_FOV = -30
#FRM_PTS_NUM = int(PTS_PER_SEC / (SENSOR_TICK * CHA_NUM))

# depth sensor parameters
SENSOR_TYPE_3 = 'CameraDepth'
SENSOR_TYPE_3_CARLA = 'sensor.camera.depth'
DEPTH_FOV = 120
DEPTH_SENSOR_TICK = 0.0

# albedo sensor parameters
SENSOR_TYPE_6 = 'RealCam'
SENSOR_TYPE_6_CARLA = 'sensor.camera.rgb'
RGB_FOV = 120
RGB_SENSOR_TICK = 0.0

# create folders for storing data


if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"depth/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"depth/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"semantic/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"semantic/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"rgb/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"rgb/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"left/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"left/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"right/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"right/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME+"/"+"rear/"):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"rear/")

if not os.path.exists(SAVE_PATH+NAME_WITH_TIME):
    os.makedirs(SAVE_PATH+NAME_WITH_TIME+"/"+"Lidar/")


def main():
    actor_list = []
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(2.0)

    world = client.get_world() #get_world()  ### new
    print('enabling synchronous mode.')

    try:
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1 ### new
        settings.synchronous_mode = True
        world.apply_settings(settings)

        blueprints = world.get_blueprint_library()
        vehicle_blueprint = blueprints.filter("vehicle.*")
        mycar_blueprint = blueprints.filter("vehicle.audi*")
        pedestrain_blueprint = blueprints.filter("walker.*")

        camera_seg_blueprint = blueprints.find(SENSOR_TYPE_1_CARLA)
        camera_seg_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_seg_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_seg_blueprint.set_attribute('fov', str(FOV))
        camera_seg_blueprint.set_attribute('sensor_tick', str(SENSOR_TICK))

        lidar_blueprint = blueprints.find(SENSOR_TYPE_2_CARLA)
        lidar_blueprint.set_attribute('channels', str(CHA_NUM))
        lidar_blueprint.set_attribute('range', str(RANGE))
        lidar_blueprint.set_attribute('points_per_second', str(PTS_PER_SEC))
        lidar_blueprint.set_attribute('rotation_frequency', str(ROT_FREQ))
        lidar_blueprint.set_attribute('upper_fov', str(UPPER_FOV))
        lidar_blueprint.set_attribute('lower_fov', str(LOWER_FOV))
        lidar_blueprint.set_attribute('sensor_tick', str(SENSOR_TICK))

        depth_blueprint = blueprints.find(SENSOR_TYPE_3_CARLA)
        depth_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        depth_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        depth_blueprint.set_attribute('fov', str(DEPTH_FOV))
        depth_blueprint.set_attribute('sensor_tick', str(DEPTH_SENSOR_TICK))

        rgb_blueprint = blueprints.find(SENSOR_TYPE_6_CARLA)
        rgb_blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        rgb_blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        rgb_blueprint.set_attribute('fov', str(RGB_FOV))
        rgb_blueprint.set_attribute('sensor_tick', str(RGB_SENSOR_TICK))

        m = world.get_map()
        spawn_points = m.get_spawn_points()

        random.shuffle(spawn_points)

        # test whether vehicle number exceeds spawn points number
        if len(spawn_points) >= OTHER_VEH_NUM:
            veh_num = OTHER_VEH_NUM
            rest = len(spawn_points) - OTHER_VEH_NUM
            if rest >= OTHER_PED_NUM:
                ped_num = OTHER_PED_NUM
            else:
                ped_num = rest
                print("Do not have so many spawn points. %d vehicles and %d pedestrains created." % (veh_num, ped_num))
        else:
            veh_num = len(spawn_points)
            print("Do not have so many spawn points. %d vehicles and %d pedestrains created." % (veh_num, 0))

        for i in range(veh_num):
            other_veh = world.spawn_actor(random.choice(vehicle_blueprint), spawn_points[i])
            print(other_veh.attributes)
            other_veh.set_autopilot(1)
            actor_list.append(other_veh)

        for i in range(OTHER_PED_NUM):
            ped = world.spawn_actor(random.choice(pedestrain_blueprint), spawn_points[i+OTHER_VEH_NUM])
            print(ped.attributes)
            player_control = carla.WalkerControl()
            player_control.speed = 3
            pedestrian_heading = 90
            player_rotation = carla.Rotation(0, pedestrian_heading, 0)
            player_control.direction = player_rotation.get_forward_vector()
            ped.apply_control(player_control)
            actor_list.append(ped)

        # create my own car
        my_car = world.spawn_actor(random.choice(mycar_blueprint), random.choice(spawn_points))
        waypoint = m.get_waypoint(random.choice(spawn_points).location)
        print(f'car attributes: {my_car.attributes}')
        my_car.set_autopilot(1)

        transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        transform_front = carla.Transform(carla.Location(x=1.6, z=1.7))

        # create sensors and attach them to my_car
        depth_front = world.spawn_actor(depth_blueprint, transform_front, attach_to=my_car)
        print(f'depth front attributes: {depth_front.attributes}')

        # create sensors and attach them to my_car
        lidar_top = world.spawn_actor(lidar_blueprint, transform, attach_to=my_car)
        print(f'lidar top attributes: {lidar_top.attributes}')

        semantic_front = world.spawn_actor(camera_seg_blueprint, transform_front, attach_to=my_car)
        print(f'semantic front attributes: {semantic_front.attributes}')

        rgb_front = world.spawn_actor(rgb_blueprint, transform_front, attach_to=my_car)
        print(f'rgb front attributes: {rgb_front.attributes}')

        # Make sync queue for sensor data.
        image_queue_1 = queue.Queue()
        image_queue_2 = queue.Queue()
        image_queue_4 = queue.Queue()
        image_queue_5 = queue.Queue()

        depth_front.listen(image_queue_1.put)
        lidar_top.listen(image_queue_2.put)
        semantic_front.listen(image_queue_4.put)
        rgb_front.listen(image_queue_5.put)

        actor_list.append(depth_front)
        actor_list.append(semantic_front)
        actor_list.append(rgb_front)

        actor_list.append(my_car)

        pygame.init()
        display = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        font = get_font()

        clock = pygame.time.Clock()
        # begin to record time for stopping
        counter = 0

        while True:
            if should_quit():
                return
            world.tick()
            clock.tick()
            counter += 1
            print(f'Frame: {counter}')
            
            # Choose the next waypoint and update the car location.
            waypoint = random.choice(waypoint.next(1.5))
            my_car.set_transform(waypoint.transform)

            # ts = world.wait_for_tick()

            image_depth = image_queue_1.get()
            image_depth.convert(carla.ColorConverter.LogarithmicDepth)
            image_depth.save_to_disk(
                SAVE_PATH+NAME_WITH_TIME+"/"+"depth/%06d.png" % image_depth.frame_number)

            image_lidar = image_queue_2.get()
            image_lidar.save_to_disk(
                SAVE_PATH+NAME_WITH_TIME+"/"+"Lidar/%06d.ply" % image_lidar.frame_number)

            image_semantic = image_queue_4.get()
            image_semantic.convert(carla.ColorConverter.CityScapesPalette)
            image_semantic.save_to_disk(
                SAVE_PATH+NAME_WITH_TIME+"/"+"semantic/%06d.png" % image_semantic.frame_number)

            image_rgb = image_queue_5.get()
            image_rgb.save_to_disk(
                SAVE_PATH+NAME_WITH_TIME+"/"+"rgb/%06d.png" % image_rgb.frame_number)

            if(counter >= STOP_AFTER):
                break
            # Draw the display.
            draw_image(display, image_rgb)
            draw_image(display, image_semantic, blend=True)
            display.blit(
                font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                (8, 10))
            pygame.display.flip()
    
    finally:

        for actor in actor_list:
            id = actor.id
            actor.destroy()
            print("Actor %d destroyed.\n" % id)

        print('\ndisabling synchronous mode.')
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')