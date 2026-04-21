# Credit goes to https://www.kaggle.com/code/kalikichandu/preprossing-inkml-to-png-files for original code.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
import xml.etree.ElementTree as ET
import os
import numpy as np
from tqdm import tqdm
import cv2
import collections


def get_traces_data(inkml_file_abs_path):

    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"

    # Extract all traces and their corresponding ids using a standard readable loop
    traces_all = []
    for trace_tag in root.findall(doc_namespace + "trace"):
        trace_id = trace_tag.get("id")

        # Parse the coordinate string for this trace
        raw_coords = trace_tag.text.replace("\n", "").split(",")
        parsed_coords = []

        for coord_str in raw_coords:
            coord_str = coord_str.strip()
            if not coord_str:
                continue

            point = []
            for axis_coord in coord_str.split(" "):
                val = float(axis_coord)
                # If the coordinate is a decimal, scale it up to preserve precision as an integer
                if not val.is_integer():
                    val = val * 10000
                point.append(round(val))

            parsed_coords.append(point)

        traces_all.append({"id": trace_id, "coords": parsed_coords})

    #   'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict["id"]))

    #   'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + "traceGroup")

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + "traceGroup"):

            label = traceGroup.find(doc_namespace + "annotation").text

            #    'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + "traceView"):

                #     'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get("traceDataRef"))

                #     'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]["coords"]
                traces_curr.append(single_trace)

            traces_data.append({"label": label, "trace_group": traces_curr})

    else:
        #             'Consider Validation data that has no labels'
        [traces_data.append({"trace_group": [trace["coords"]]}) for trace in traces_all]
    print(traces_data)
    return traces_data


def get_gt(inkml_file_abs_path):
    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    annotation = root.find(f".//{doc_namespace}annotation[@type='truth']")
    if annotation is not None:
        truth = annotation.text
    else:
        raise Exception("No truth annotation found.")
    return truth


def inkml2img(input_path, output_path):
    traces = get_traces_data(input_path)

    # Configure matplotlib to draw just the lines without any axes or borders
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # Draw each trace onto the plot
    for elem in traces:
        trace_group = elem["trace_group"]
        for sub_trace in trace_group:
            data = np.array(sub_trace)

            # We only need X and Y coordinates; ignore pressure/tilt if present
            if data.shape[1] > 2:
                data = data[:, :2]

            x, y = zip(*data)
            plt.plot(x, y, linewidth=2, c="black")

    # Ensure output directory exists (exist_ok prevents errors if it's already there)
    os.makedirs(output_path, exist_ok=True)

    # Generate a safe, flat filename
    input_path_safe = (
        os.path.normpath(input_path)
        .replace(os.sep, "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    file_name = 0
    base_output_name = input_path_safe.replace(".", "_").replace("_inkml", ".inkml")

    save_path = os.path.join(output_path, f"{base_output_name}_{file_name}.png")

    if os.path.isfile(save_path):
        file_name += 1
        save_path = os.path.join(output_path, f"{base_output_name}_{file_name}.png")

    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=100,
    )
    plt.gcf().clear()


def ink2img_folder(input_paths, output_path):
    labels = collections.defaultdict(list)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for input_path in input_paths:
        files = os.listdir(input_path)
        # ignore all files that don't have the .inkML extension
        files = [file for file in files if file.endswith(".inkml")]
        for file in tqdm(files):
            inkML_path = os.path.join(input_path, file)
            # Create the same safe name as in inkml2img
            input_path_safe = (
                os.path.normpath(inkML_path)
                .replace(os.sep, "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            image_name = (
                input_path_safe.replace(".", "_").replace("_inkml", ".inkml") + "_0.png"
            )

            try:
                labels["label"].append(get_gt(inkML_path))
                labels["name"].append(image_name)
                inkml2img(inkML_path, output_path)
            except Exception as e:
                print(e)
                print(
                    "Error with file: "
                    + str(file)
                    + " in folder: "
                    + str(input_path)
                    + ". Don't worry, this is expected (though there should only be max 2 or 3!)."
                )
    pd.DataFrame(labels).to_csv(os.path.join(output_path, "labels.csv"), index=False)
