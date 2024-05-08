import numpy as np
from PIL import Image, ImageOps
import math


def barycentric(x, y, x0, y0, x1, y1, x2, y2):

    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

    if not denominator:
        return -1, -1, -1

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
    lambda2 = 1.0 - lambda0 - lambda1

    return lambda0, lambda1, lambda2


def perspective(x, y, z):
    coordinates_eq = np.array([x / z, y / z, 1])
    scale = np.array([[1000, 0, 1000], [0, 1000, 1000], [0, 0, 1]])
    persp_coordinates = scale @ coordinates_eq
    return persp_coordinates[0], persp_coordinates[1]


def drawtriangle(
    x0,
    y0,
    z0,
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    cosI0,
    cosI1,
    cosI2,
    texture_pic,
    vt_0,
    vt_1,
    vt_2,
):

    p_x0, p_y0 = perspective(x0, y0, z0)
    p_x1, p_y1 = perspective(x1, y1, z1)
    p_x2, p_y2 = perspective(x2, y2, z2)

    xmin = int(min(p_x0, p_x1, p_x2))
    xmax = int(max(p_x0, p_x1, p_x2))
    ymin = int(min(p_y0, p_y1, p_y2))
    ymax = int(max(p_y0, p_y1, p_y2))

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0

    for u in range(xmin, xmax + 1):
        for v in range(ymin, ymax + 1):
            lambda0, lambda1, lambda2 = barycentric(
                u, v, p_x0, p_y0, p_x1, p_y1, p_x2, p_y2
            )
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                tempZ = z0 * lambda0 + z1 * lambda1 + z2 * lambda2
                I = cosI0 * lambda0 + cosI1 * lambda1 + cosI2 * lambda2
                if I > 0:
                    I = 0
                color_test = texture_pic[
                    int(
                        1024
                        * (vt_0[1] * lambda0 + vt_1[1] * lambda1 + vt_2[1] * lambda2)
                    )
                ][
                    int(
                        1024
                        * (vt_0[0] * lambda0 + vt_1[0] * lambda1 + vt_2[0] * lambda2)
                    )
                ]
                color = -255 * color_test
                if tempZ < z_buf[u][v]:
                    image3d[v, u] = color
                    z_buf[u][v] = tempZ


def calculate_normal_polygon(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    v1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    v2 = np.array([x1 - x0, y1 - y0, z1 - z0])
    normal = np.cross(v1, v2)
    return normal


def calculate_light(normal, l):
    return np.dot(normal, l) / (np.linalg.norm(normal) * np.linalg.norm(l))


def cosA(normal):
    return np.dot(normal, (0, 0, 1)) / np.linalg.norm(normal)


def transform(x, y, z):
    coordinates_old = np.array([x, y, z])
    scale = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    shift = np.array([0, -0.04, 0.1])
    alpha, beta, gamma = 0, (5 * math.pi) / 4, 0
    r_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), np.sin(alpha)],
            [0, -np.sin(alpha), np.cos(alpha)],
        ]
    )
    r_y = np.array(
        [
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)],
        ]
    )
    r_z = np.array(
        [
            [np.cos(gamma), np.sin(gamma), 0],
            [-np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    coordinates_new = r_x @ r_y @ r_z @ scale @ coordinates_old + shift
    return coordinates_new[0], coordinates_new[1], coordinates_new[2]


image3d = np.full((2000, 2000, 3), (75, 0, 130), dtype=np.uint8)
vertex_list = []
edge_list = []
polygons_list = []
texture_list = []
texture_nums = []
vt_list = []

f = open("model_1.obj")

for s in f:
    splitted = s.split()
    if splitted[0] == "v":
        vertex_list.append([float(x) for x in splitted[1:4]])
    if splitted[0] == "f":
        for i in splitted[1:4]:
            edge_list.append(int(i.split("/")[0]))
            texture_nums.append(int(i.split("/")[1]))
    if splitted[0] == "vt":
        texture_list.append([float(x) for x in splitted[1:3]])


edge_list = [edge_list[i : i + 3] for i in range(0, len(edge_list), 3)]
texture_nums = [texture_nums[i : i + 3] for i in range(0, len(texture_nums), 3)]

vertex_normals = np.zeros((len(vertex_list), 3))

z_buf = np.full((2000, 2000), np.inf)

for j in range(len(edge_list)):
    x0, y0, z0 = transform(
        (vertex_list[edge_list[j][0] - 1][0]),
        (vertex_list[edge_list[j][0] - 1][1]),
        (vertex_list[edge_list[j][0] - 1][2]),
    )
    x1, y1, z1 = transform(
        (vertex_list[edge_list[j][1] - 1][0]),
        (vertex_list[edge_list[j][1] - 1][1]),
        (vertex_list[edge_list[j][1] - 1][2]),
    )
    x2, y2, z2 = transform(
        (vertex_list[edge_list[j][2] - 1][0]),
        (vertex_list[edge_list[j][2] - 1][1]),
        (vertex_list[edge_list[j][2] - 1][2]),
    )

    vt_0, vt_1, vt_2 = (
        [
            texture_list[texture_nums[j][0] - 1][0],
            texture_list[texture_nums[j][0] - 1][1],
        ],
        [
            texture_list[texture_nums[j][1] - 1][0],
            texture_list[texture_nums[j][1] - 1][1],
        ],
        [
            texture_list[texture_nums[j][2] - 1][0],
            texture_list[texture_nums[j][2] - 1][1],
        ],
    )

    vt_list.append([vt_0, vt_1, vt_2])
    polygons_list.append([x0, y0, z0, x1, y1, z1, x2, y2, z2])

    polygons_normals = calculate_normal_polygon(x0, y0, z0, x1, y1, z1, x2, y2, z2)

    vertex_normals[edge_list[j][0] - 1] += polygons_normals
    vertex_normals[edge_list[j][1] - 1] += polygons_normals
    vertex_normals[edge_list[j][2] - 1] += polygons_normals

for i in range(len(vertex_normals)):
    vertex_normals[i] /= np.linalg.norm(vertex_normals[i])

light_list = []

for i in range(len(vertex_normals)):
    light_list.append(calculate_light(vertex_normals[i], np.array([0, 0, 1])))

texture_pic = np.array(ImageOps.flip(Image.open("bunny.jpg")))

j = 0
for i in polygons_list:
    if (
        cosA(
            calculate_normal_polygon(
                i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]
            )
        )
        < 0
    ):
        drawtriangle(
            i[0],
            i[1],
            i[2],
            i[3],
            i[4],
            i[5],
            i[6],
            i[7],
            i[8],
            light_list[edge_list[j][0] - 1],
            light_list[edge_list[j][1] - 1],
            light_list[edge_list[j][2] - 1],
            texture_pic,
            vt_list[j][0],
            vt_list[j][1],
            vt_list[j][2],
        )
    j += 1

img = Image.fromarray(image3d, mode="RGB")
img = ImageOps.flip(img)
img.show()
img.save("6.png")
