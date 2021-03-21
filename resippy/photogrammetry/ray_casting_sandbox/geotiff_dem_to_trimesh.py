import time
import trimesh


def geotiff_dem_to_trimesh(geotiff_image):
    # mesh objects can be created from existing faces and vertex data
    image_data = geotiff_image.get_image_band(0)
    facets = []
    facet_ids = []
    ny, nx = image_data.shape
    facet_counter = 0
    for x in range(nx - 1):
        for y in range(ny - 1):
            # create facet for first triangle in DEM pixel
            vertex_1 = image_data[y, x]
            vertex_2 = image_data[y+1, x+1]
            vertex_3 = image_data[y+1, x]
            facets.append([vertex_1, vertex_2, vertex_3])
            facet_ids.append(facet_counter)
            facet_counter += 1

            # create facet for second triangle in DEM pixel
            vertex_1 = image_data[y, x]
            vertex_2 = image_data[y+1, x+1]
            vertex_3 = image_data[y, x+1]
            facets.append([vertex_1, vertex_2, vertex_3])
            facet_ids.append(facet_counter)
            facet_counter += 1
    tic = time.time()
    mesh = trimesh.Trimesh(vertices=facets,
                           faces=[facet_ids])
    toc = time.time()
    print(toc-tic)

    return mesh
