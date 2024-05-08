import numpy as np
from PIL import Image, ImageOps


def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

    if not denominator:
        return -1, -1, -1

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


def drawtriangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, cosA):
    xmin = min(x0, x1, x2)
    xmax = max(x0, x1, x2)
    ymin = min(y0, y1, y2)
    ymax = max(y0, y1, y2)

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0

    color = -255 * cosA

    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            lambda0, lambda1, lambda2 = barycentric(x, y, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                tempZ = z0 * lambda0 + z1 * lambda1 + z2 * lambda2
                if tempZ < z_buf[x][y]:
                    image3d[y, x] = color
                    z_buf[x][y] = tempZ


def calculate_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    v1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    v2 = np.array([x1 - x0, y1 - y0, z1 - z0])
    normal = np.cross(v1, v2)
    return normal


def cosA(normal):
    return np.dot(normal, (0, 0, 1)) / np.linalg.norm(normal)


image3d = np.full((2000, 2000, 3), (75, 0, 130), dtype=np.uint8)
vertex_list = []
edge_list = []

f = open("model_1.obj")

for s in f:
    splitted = s.split()
    if splitted[0] == "v":
        vertex_list.append([float(x) for x in splitted[1:4]])
    if splitted[0] == "f":
        for i in splitted[1:4]:
            edge_list.append(int(i.split("/")[0]))

edge_list = [edge_list[i: i + 3] for i in range(0, len(edge_list), 3)]

z_buf = np.full((2000, 2000), np.inf)

for j in range(len(edge_list)):
    x0 = int(vertex_list[edge_list[j][0] - 1][0] * 10000) + 1000
    y0 = int(vertex_list[edge_list[j][0] - 1][1] * 10000) + 600
    z0 = int(vertex_list[edge_list[j][0] - 1][2] * 10000)
    x1 = int(vertex_list[edge_list[j][1] - 1][0] * 10000) + 1000
    y1 = int(vertex_list[edge_list[j][1] - 1][1] * 10000) + 600
    z1 = int(vertex_list[edge_list[j][1] - 1][2] * 10000)
    x2 = int(vertex_list[edge_list[j][2] - 1][0] * 10000) + 1000
    y2 = int(vertex_list[edge_list[j][2] - 1][1] * 10000) + 600
    z2 = int(vertex_list[edge_list[j][2] - 1][2] * 10000)

    if cosA(calculate_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)) < 0:
        drawtriangle(x0, y0, z0, x1, y1, z1, x2, y2, z2, cosA(calculate_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)))
    else:
        continue

img = Image.fromarray(image3d, mode="RGB")
img = ImageOps.flip(img)
img.show()
