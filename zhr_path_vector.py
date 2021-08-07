import sys
if '/home/u/Desktop/habitat-lab' in sys.path:
    del sys.path[sys.path.index('/home/u/Desktop/habitat-lab')]
    print("Delete habitat-lab")   
sys.path.insert(0,"/home/u/Desktop/PPO_habitat-lab")
print("Insert PPO_habitat-lab")
# if '/home/u/Desktop/habitat-sim' in sys.path:
#     del sys.path[sys.path.index('/home/u/Desktop/habitat-sim')]
#     print("Delete habitat-sim") 
# sys.path.insert(0,"/home/u/Desktop/PPO_habitat-sim")
import math
import os
import random
import sys
import git
import imageio
import magnum as mn
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut
from habitat.utils.visualizations import maps

test_scene = "/home/u/Desktop/17DRP5sb8fy/17DRP5sb8fy.glb"
rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}
sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene.id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_specs.append(sensor_spec)
    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
cfg = make_cfg(sim_settings)
try:  # Got to make initialization idiot proof
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)
# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
agent.set_state(agent_state)
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
print(sim.pathfinder.is_loaded)


def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown
# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)
    plt.pause(1)





print("NavMesh area = " + str(sim.pathfinder.navigable_area))
print("Bounds = " + str(sim.pathfinder.get_bounds()))

# @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
pathfinder_seed = 1  # @param {type:"integer"}
sim.pathfinder.seed(pathfinder_seed)
nav_point = sim.pathfinder.get_random_navigable_point()
print("Random navigable point : " + str(nav_point))
print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

# @markdown The radius of the minimum containing circle (with vertex centroid origin) for the isolated navigable island of a point can be queried with *island_radius*.
# @markdown This is analogous to the size of the point's connected component and can be used to check that a queried navigable point is on an interesting surface (e.g. the floor), rather than a small surface (e.g. a table-top).
print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))

# @markdown The closest boundary point can also be queried (within some radius).
max_search_radius = 2.0  # @param {type:"number"}
print(
    "Distance to obstacle: "
    + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
)
hit_record = sim.pathfinder.closest_obstacle_surface_point(
    nav_point, max_search_radius
)
print("Closest obstacle HitRecord:")
print(" point: " + str(hit_record.hit_pos))
print(" normal: " + str(hit_record.hit_normal))
print(" distance: " + str(hit_record.hit_dist))

vis_points = [nav_point]

# HitRecord will have infinite distance if no valid point was found:
if math.isinf(hit_record.hit_dist):
    print("No obstacle found within search radius.")
else:
    # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
    perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
    print("Perturbed point : " + str(perturbed_point))
    print(
        "Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
    )
    snapped_point = sim.pathfinder.snap_point(perturbed_point)
    print("Snapped point : " + str(snapped_point))
    print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
    vis_points.append(snapped_point)

# @markdown ---
# @markdown ### Visualization
# @markdown Running this cell generates a topdown visualization of the NavMesh with sampled points overlayed.
meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}

display =True
if display:
    xy_vis_points = convert_points_to_topdown(
        sim.pathfinder, vis_points, meters_per_pixel
    )
    # use the y coordinate of the sampled nav_point for the map height slice
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    print("\nDisplay the map with key_point overlay:")
    display_map(top_down_map, key_points=xy_vis_points)


print("++++++++++\n"*10)

seed = 4  # @param {type:"integer"}
sim.pathfinder.seed(seed)

# fmt off
# @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
# fmt on
sample1 = sim.pathfinder.get_random_navigable_point()
sample2 = sim.pathfinder.get_random_navigable_point()

# @markdown 2. Use ShortestPath module to compute path between samples.
path = habitat_sim.ShortestPath()
path.requested_start = sample1
path.requested_end = sample2
found_path = sim.pathfinder.find_path(path)
geodesic_distance = path.geodesic_distance
path_points = path.points
# @markdown - Success, geodesic path length, and 3D points can be queried.
print("found_path : " + str(found_path))
print("geodesic_distance : " + str(geodesic_distance))
print("path_points : " + str(path_points))

# @markdown 3. Display trajectory (if found) on a topdown map of ground floor
if found_path:
    meters_per_pixel = 0.025
    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    height = scene_bb.y().min
    display = True
    if display:
        top_down_map = maps.get_topdown_map(
            sim.pathfinder, height, meters_per_pixel=meters_per_pixel
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
        # convert world trajectory points to maps module grid points
        trajectory = [
            maps.to_grid(
                path_point[2],
                path_point[0],
                grid_dimensions,
                pathfinder=sim.pathfinder,
            )
            for path_point in path_points
        ]
        grid_tangent = mn.Vector2(
            trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
        )
        path_initial_tangent = grid_tangent / grid_tangent.length()
        initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
        # draw the agent and trajectory on the map
        maps.draw_path(top_down_map, trajectory)
        maps.draw_agent(
            top_down_map, trajectory[0], initial_angle, agent_radius_px=8
        )
        print("\nDisplay the map with agent and path overlay:")
        display_map(top_down_map)

    # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
    display_path_agent_renders = True  # @param{type:"boolean"}
    if display_path_agent_renders:
        print("Rendering observations at path points:")
        tangent = path_points[1] - path_points[0]
        agent_state = habitat_sim.AgentState()
        for ix, point in enumerate(path_points):
            if ix < len(path_points) - 1:
                tangent = path_points[ix + 1] - point
                agent_state.position = point
                tangent_orientation_matrix = mn.Matrix4.look_at(
                    point, point + tangent, np.array([0, 1.0, 0])
                )
                tangent_orientation_q = mn.Quaternion.from_matrix(
                    tangent_orientation_matrix.rotation()
                )
                agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                agent.set_state(agent_state)

                observations = sim.get_sensor_observations()
                rgb = observations["color_sensor"]
                semantic = observations["semantic_sensor"]
                depth = observations["depth_sensor"]

                if display:
                    def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
                        from habitat_sim.utils.common import d3_40_colors_rgb

                        rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

                        arr = [rgb_img]
                        titles = ["rgb"]
                        if semantic_obs.size != 0:
                            semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
                            semantic_img.putpalette(d3_40_colors_rgb.flatten())
                            semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
                            semantic_img = semantic_img.convert("RGBA")
                            arr.append(semantic_img)
                            titles.append("semantic")

                        if depth_obs.size != 0:
                            depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
                            arr.append(depth_img)
                            titles.append("depth")

                        plt.figure(figsize=(12, 8))
                        for i, data in enumerate(arr):
                            ax = plt.subplot(1, 3, i + 1)
                            ax.axis("off")
                            ax.set_title(titles[i])
                            plt.imshow(data)
                        plt.show(block=False)
                        plt.pause(1)
                    display_sample(rgb, semantic, depth)
