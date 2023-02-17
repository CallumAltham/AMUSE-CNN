from shapely.geometry import LineString, MultiPolygon, Point
import numpy as np


def moments(geom):
    centroid_x, centroid_y = geom.centroid.coords[0]
    geom_x, geom_y = geom.exterior.xy
    geom_x, geom_y = np.array(geom_x) - centroid_x, np.array(geom_y) - centroid_y
    if geom_x[0] != geom_x[-1] or geom_y[0] != geom_y[-1]:  # Make sure the first vertex is also the last vertex
        geom_x, geom_y = np.hstack((geom_x, geom_x[0])), np.hstack((geom_y, geom_y[0]))
    x_i_y_i = geom_x[:-1] * geom_y[:-1]
    x_i_y_i1 = geom_x[:-1] * geom_y[1:]
    x_i1_y_i = geom_x[1:] * geom_y[:-1]
    x_i1_y_i1 = geom_x[1:] * geom_y[1:]
    x_i_x_i = geom_x[:-1] ** 2
    x_i_x_i1 = geom_x[1:] * geom_x[:-1]
    x_i1_x_i1 = geom_x[1:] ** 2
    y_i_y_i = geom_y[:-1] ** 2
    y_i_y_i1 = geom_y[1:] * geom_y[:-1]
    y_i1_y_i1 = geom_y[1:] ** 2

    common_component = x_i_y_i1 - x_i1_y_i
    sm_xx = sum(common_component * (y_i_y_i + y_i_y_i1 + y_i1_y_i1)) / 12
    sm_yy = sum(common_component * (x_i_x_i + x_i_x_i1 + x_i1_x_i1)) / 12
    sm_xy = sum(common_component * (x_i_y_i1 + 2 * x_i_y_i + 2 * x_i1_y_i1 + x_i1_y_i)) / 12
    if geom.exterior.is_ccw:
        return sm_xx, sm_yy, sm_xy
    else:
        return -sm_xx, -sm_yy, -sm_xy


def theta_mb(geometry):
    sm_xx, sm_yy, sm_xy = moments(geometry)
    return 0.5 * np.arctan2(2 * sm_xy, sm_yy - sm_xx)


def segment_centroids(geometry, window_size, resolution=0.5):
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if isinstance(geometry, MultiPolygon):
        geometry = geometry[1].union(geometry[0].union(geometry[0].intersection(geometry[1]).buffer(0.001)))
    theta = theta_mb(geometry)
    x_min, y_min, x_max, y_max = geometry.bounds
    hypotenuse = ((y_max - y_min) ** 2 + (x_max - x_min) ** 2) ** 0.5
    r = np.arange(-hypotenuse, hypotenuse, step=hypotenuse / 10)
    cx, cy = geometry.centroid.coords[0]
    line_x = cx + r * np.cos(theta)
    line_y = cy + r * np.sin(theta)
    line = LineString([(x, y) for x, y in zip(line_x, line_y)])
    points = line.intersection(geometry.exterior)
    points = [p if type(p) == Point else Point(p.xy[0][-1], p.xy[1][-1]) for p in points]
    point_a, point_b = points[0], points[-1]  # first and last intersections represent the extremities along mb axis
    # logarithmic relationship between segment size and how many windows should be used
    numb_points = round((7 - 6 * np.exp(-point_a.distance(point_b) / (window_size * resolution) / 6)))
    if numb_points < 1:
        numb_points = 1
    elif numb_points % 2 == 0:
        numb_points -= 1
    points_x = np.linspace(point_a.x + (point_b.x - point_a.x) / (numb_points + 1),
                           point_b.x - (point_b.x - point_a.x) / (numb_points + 1), numb_points, endpoint=True)
    points_y = np.linspace(point_a.y + (point_b.y - point_a.y) / (numb_points + 1),
                           point_b.y - (point_b.y - point_a.y) / (numb_points + 1), numb_points, endpoint=True)

    window_centroids = []
    for x, y in zip(points_x, points_y):  # for each perpendicular line
        line_x = x + r * np.cos(theta + np.pi / 2)
        line_y = y + r * np.sin(theta + np.pi / 2)
        line = LineString([(a, b) for a, b in zip(line_x, line_y)])
        points = line.intersection(geometry.exterior)  # find line's intersections with segment
        points = [p if type(p) == Point else Point(p.xy[0][-1], p.xy[1][-1]) for p in points]
        points = [(points[2 * it], points[2 * it + 1]) for it in range(int(len(points) / 2))]
        distances = np.zeros(len(points))
        for it in range(len(points)):
            distances[it] = points[it][0].distance(points[it][1])
        max_it = np.where(distances == np.max(distances))[0][0]  # use pair with maximum separation
        point_a, point_b = points[max_it][0], points[max_it][1]
        try:
            window_centroids.append(((point_a.x + point_b.x) / 2, (point_a.y + point_b.y) / 2))
        except AttributeError:
            print(type(line.intersection(geometry.exterior)))
            # print([line.intersection(geometry.exterior))
            print([type(blah) for p in points for blah in p])
            exit(0)
    return window_centroids
