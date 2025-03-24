from .run import MatAnyoneVideo, SolidColorBatched

NODE_CLASS_MAPPINGS = {
    "MatAnyone": MatAnyoneVideo,
    "SolidColorBatched": SolidColorBatched,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyone": "MatAnyone",
    "SolidColorBatched": "Solid Color Batched",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
