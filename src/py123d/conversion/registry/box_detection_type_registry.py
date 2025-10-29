BOX_DETECTION_TYPE_REGISTRY = {}


def register_box_detection_type(enum_class):
    BOX_DETECTION_TYPE_REGISTRY[enum_class.__name__] = enum_class
    return enum_class
