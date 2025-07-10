from ultralytics import YOLO

model = YOLO("yolo8_19_UL-New_day2_fullLabTopCam_60epochs.pt")
model.export(format="imx")
