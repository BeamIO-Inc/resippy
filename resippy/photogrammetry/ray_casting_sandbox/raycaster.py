import numpy


class RayCaster:
    def __init__(self):
        self._ray_starts = None
        self._ray_ends = None

    @property
    def ray_starts(self):
        return self._ray_starts

    @ray_starts.setter
    def ray_starts(self, val):
        self._ray_starts = val

    @property
    def ray_ends(self):
        return self._ray_ends

    @ray_ends.setter
    def ray_ends(self, val):
        self._ray_ends = val

    def rays_xy_planes_intersections(self, z_heights):
        xy_intersections = numpy.zeros((self.ray_starts.shape[0], 2))
        xy_intersections[:, 0] = (self.ray_ends[:, 0] - self.ray_starts[:, 0]) * \
                                 (z_heights - self.ray_starts[:, 2]) / \
                                 (self.ray_ends[:, 2] - self.ray_starts[:, 2])
        xy_intersections[:, 1] = (self.ray_ends[:, 1] - self.ray_starts[:, 1]) * \
                                 (z_heights - self.ray_starts[:, 2]) / \
                                 (self.ray_ends[:, 2] - self.ray_starts[:, 2])
        return xy_intersections

    def rays_xz_planes_intersections(self, y_locs):
        xz_intersections = numpy.zeros((self.ray_starts.shape[0], 2))
        xz_intersections[:, 0] = (self.ray_ends[:, 0] - self.ray_starts[:, 0]) * \
                                 (y_locs - self.ray_starts[:, 1]) / \
                                 (self.ray_ends[:, 1] - self.ray_starts[:, 1])
        xz_intersections[:, 1] = (self.ray_ends[:, 2] - self.ray_starts[:, 2]) * \
                                 (y_locs - self.ray_starts[:, 1]) / \
                                 (self.ray_ends[:, 1] - self.ray_starts[:, 1])
        return xz_intersections

    def rays_yz_planes_intersections(self, x_locs):
        yz_intersections = numpy.zeros((self.ray_starts.shape[0], 2))
        yz_intersections[:, 0] = (self.ray_ends[:, 1] - self.ray_starts[:, 1]) * \
                                 (x_locs - self.ray_starts[:, 0]) / \
                                 (self.ray_ends[:, 0] - self.ray_starts[:, 0])
        yz_intersections[:, 1] = (self.ray_ends[:, 2] - self.ray_starts[:, 2]) * \
                                 (x_locs - self.ray_starts[:, 0]) / \
                                 (self.ray_ends[:, 0] - self.ray_starts[:, 0])
        return yz_intersections

    def rays_intersect_boxes(self,
                             boxes_x_mins,  # type: float
                             boxes_x_maxes,  # type: float
                             boxes_y_mins,  # type: float
                             boxes_y_maxes,  # type: float
                             boxes_z_mins,  # type: float
                             boxes_z_maxes,  # type: float
                             ):
        top_intersections = self.rays_xy_planes_intersections(boxes_z_maxes)
        bottom_intersections = self.rays_xy_planes_intersections(boxes_z_mins)
        right_intersections = self.rays_xz_planes_intersections(boxes_y_maxes)
        left_intersections = self.rays_xz_planes_intersections(boxes_y_mins)
        front_intersections = self.rays_yz_planes_intersections(boxes_x_maxes)
        back_intersections = self.rays_yz_planes_intersections(boxes_x_mins)

        rays_hit_tops = (top_intersections[:, 0] <= boxes_x_maxes) * \
                        (top_intersections[:, 0] >= boxes_x_mins) * \
                        (top_intersections[:, 1] <= boxes_y_maxes) * \
                        (top_intersections[:, 1] >= boxes_y_mins)
        rays_hit_bottoms = (bottom_intersections[:, 0] <= boxes_x_maxes) * \
                           (bottom_intersections[:, 0] >= boxes_x_mins) * \
                           (bottom_intersections[:, 1] <= boxes_y_maxes) * \
                           (bottom_intersections[:, 1] >= boxes_y_mins)
        rays_hit_right_sides = (right_intersections[:, 0] <= boxes_x_maxes) * \
                               (right_intersections[:, 0] >= boxes_x_mins) * \
                               (right_intersections[:, 1] <= boxes_z_maxes) * \
                               (right_intersections[:, 1] >= boxes_z_mins)
        rays_hit_left_sides = (left_intersections[:, 0] <= boxes_x_maxes) * \
                              (left_intersections[:, 0] >= boxes_x_mins) * \
                              (left_intersections[:, 1] <= boxes_z_maxes) * \
                              (left_intersections[:, 1] >= boxes_z_mins)
        rays_hit_fronts = (front_intersections[:, 0] <= boxes_y_maxes) * \
                          (front_intersections[:, 0] >= boxes_y_mins) * \
                          (front_intersections[:, 1] <= boxes_z_maxes) * \
                          (front_intersections[:, 1] >= boxes_z_mins)
        rays_hit_backs = (back_intersections[:, 0] <= boxes_y_maxes) * \
                         (back_intersections[:, 0] >= boxes_y_mins) * \
                         (back_intersections[:, 1] <= boxes_z_maxes) * \
                         (back_intersections[:, 1] >= boxes_z_mins)
        stop = 1
