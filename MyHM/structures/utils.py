import numpy as _np
from scipy.spatial.distance import cdist

def bcube_from_bbox(bbox):
    min_value , max_value = _np.min(bbox), _np.max(bbox)
    return _np.array([[min_value, max_value]] * 3)

def tuple2index(t):
    id_x, id_y, id_z = t
    return id_x + id_y * 2 + id_z * 4

def index2tuple(i):
    id_z = i // 4
    id_y = (i % 4) // 2
    id_x = ((i % 4) % 2)
    return (id_x, id_y, id_z)

def admissibility(bbox1, bbox2):
    # Diam:
    diam = _np.max(_np.linalg.norm([bbox1[:,1] - bbox1[:,0], bbox2[:,1] - bbox2[:,0]], axis=1))
    # print(diam)

    # Dist: (min of distance between all vertices)
    vertex_bbox1 = []
    for x in bbox1[0,:]:
        for y in bbox1[1,:]:
            for z in bbox1[2,:]:
                vertex_bbox1.append([x,y,z])
    
    vertex_bbox2 = []
    for x in bbox2[0,:]:
        for y in bbox2[1,:]:
            for z in bbox2[2,:]:
                vertex_bbox2.append([x,y,z])
    dist = _np.min(cdist(vertex_bbox1, vertex_bbox2))
    # print(dist)

    return diam <= dist

if __name__ == "__main__":
    bbox = _np.array([[-91.67590332, -34.7798996 ],
                      [-78.86689758,  51.68640137],
                      [-26.42860031,  45.42950058]])
    bcube_from_bbox(bbox)
    index2tuple(0)
    admissibility(bbox, bbox)
    print("Working...")