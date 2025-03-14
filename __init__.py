from .run import MatAnyoneVideo

NODE_CLASS_MAPPINGS = {
    "MatAnyone": MatAnyoneVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyone": "MatAnyone",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
