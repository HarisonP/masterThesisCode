import math

def round_decorate_array(func):
   def func_wrapper(point_a, point_b):
       return [round(x, 2) for x in func(point_a, point_b)]
   return func_wrapper

def round_decorate(func):
   def func_wrapper(point_a, point_b):
       return round(func(point_a, point_b), 2)
   return func_wrapper

@round_decorate
def point_distance(point_a, point_b):
    return math.fabs(math.hypot(point_b[0]- point_a[0], point_b[1] - point_a[1]))

@round_decorate_array
def middle_point_between(point_a, point_b):
    return [int(middle_1dpoint(point_b[0], point_a[0])), int(middle_1dpoint(point_b[1], point_a[1]))]

@round_decorate
def middle_1dpoint(point_a, point_b):
    return (point_b + point_a) / 2

def middle_n_1dpoint(points):
    return sum(points) / len(points)


def middle_point_between_n(points):
    return [int(middle_n_1dpoint([p[0] for p in points])), int(middle_n_1dpoint([p[1] for p in points]))]