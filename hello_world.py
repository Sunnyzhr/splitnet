from zhr_path import zhr_ShortestPath
test_scene = "data/scene_datasets/gibson/Denmark.glb"
requested_start = [-1.8495619297027588, 0.16554422676563263, -0.9413928985595703]
requested_end = [1.3452529907226562, 0.16554422676563263, 2.80397891998291]
template = zhr_ShortestPath(test_scene)#zhr : it takes 1 seconds !!!
found_path, geodesic_distance, path_points = template.get_path_points(requested_start,requested_end)
x = template.get_path_points(requested_start,requested_end)
print("+++++++++++",x)
print("found_path : " + str(found_path))
print("geodesic_distance : " + str(geodesic_distance))
print("path_points : " + str(path_points))


