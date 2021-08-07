import habitat_sim
class zhr_ShortestPath(object):
    def __init__(self, scene_path):
        self.path = habitat_sim.ShortestPath()
        self.scene_path = scene_path
        self.sim_cfg = habitat_sim.SimulatorConfiguration()
        self.sim_cfg.scene.id = self.scene_path
        self.agent_cfg = habitat_sim.agent.AgentConfiguration()
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(self.sim_cfg, [self.agent_cfg])) #zhr : it takes 1 seconds !!!
    def get_path_points(self, requested_start, requested_end):    
        self.path.requested_start = requested_start
        self.path.requested_end = requested_end
        found_path = self.sim.pathfinder.find_path(self.path)
        geodesic_distance = self.path.geodesic_distance
        path_points = self.path.points
        print(dir(self.sim)) 
        return found_path, geodesic_distance, path_points
if __name__ == "__main__":
    import time
    t1 = time.time()
    test_scene = "/home/u/Desktop/17DRP5sb8fy/Eudora.glb"
    requested_start = [-1.8495619297027588, 0.16554422676563263, -0.9413928985595703]
    requested_end = [1.3452529907226562, 0.16554422676563263, 2.80397891998291]
    t2 = time.time()
    template = zhr_ShortestPath(test_scene)#zhr : it takes 1 seconds !!!
    t3 = time.time()
    found_path, geodesic_distance, path_points = template.get_path_points(requested_start,requested_end)
    t4 = time.time()
    print("found_path : " + str(found_path))
    print("geodesic_distance : " + str(geodesic_distance))
    print("path_points : " + str(path_points))
    print(t2-t1)
    print(t3-t2)
    print(t4-t3)




