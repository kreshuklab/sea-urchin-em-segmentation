import mobie.metadata as mdata

# add bookmarks for the 4 test volumes from jil
POSITIONS = [
    {"position": [96.68819639184461, 72.1844707668459, 48.2970615042884], "timepoint": 0},
    {"position": [90.40336311288618, 67.8636478875620, 45.4688865287571], "timepoint": 0},
    {"position": [88.36079229722469, 47.5950605629211, 50.2610719039629], "timepoint": 0},
    {"position": [92.99585684045653, 68.2564499674969, 45.0760844488222], "timepoint": 0}
]

metadata = mdata.read_dataset_metadata("./data/S016")
views = metadata["views"]
for pos_id, pos in enumerate(POSITIONS):
    name = f"test-volume-{pos_id}"
    viewer_transform = mdata.get_viewer_transform(position=pos["position"])
    bookmark = {"uiSelectionGroup": "bookmark", "viewerTransform": viewer_transform, "isExclusive": False}
    views[name] = bookmark
metadata["views"] = views

mdata.write_dataset_metadata("./data/S016", metadata)
