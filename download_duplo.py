from roboflow import Roboflow
rf = Roboflow(api_key="lpDsJKqB3u0n2i78Flzf")
project = rf.workspace("pose-detection-duplo").project("duplo-3.0")
dataset = project.version(23).download("yolov8")
