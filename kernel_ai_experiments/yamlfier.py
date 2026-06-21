import glob
import os
import yaml
import cv2
from sklearn.model_selection import train_test_split
import shutil
import glob
import json
import argparse


def prep_paths(base_path:str, ears_to_skip:list|None = None):
    img_pname_list = []
    maskp_list = []
    for ear in glob.glob(base_path + "/*"):
        # ensure 50 views are present
        if len(glob.glob(ear + "/*")) != 50:
            raise ValueError(f"Not 50 views in {ear}")
        if ears_to_skip and os.path.basename(ear) in ears_to_skip:
            continue
        
        for ear_view in glob.glob(ear + "/*"):
            if IGNOREUPS and "_up_" in ear_view:
                continue
            ear_view_name = os.path.basename(ear_view)

            imgp = os.path.join(ear_view, f"{ear_view_name}_img.jpg")

            if not os.path.exists(imgp):
                imgp = os.path.join(ear_view, f"{ear_view_name}_img.png")
            if not os.path.exists(imgp):
                raise FileNotFoundError(imgp)

            img_pname_list.append([ear_view_name, imgp])
            maskp_list.append([])
            maskfolder = os.path.join(ear_view, "masks")
            if not os.path.exists(maskfolder):
                raise FileNotFoundError(maskfolder)
            for mask in glob.glob(maskfolder + "/*"):
                maskp_list[-1].append(mask)

    return img_pname_list, maskp_list


def create_yaml_file(yamldest):
    if TESTSET:
        test = "test"
    else:
        test = ""

    data = {
        "path": yamldest,
        "train": "train",
        "val": "val",
        "test": f"{test}",
        "names": {
            0: "kernel",  # Assuming single class for segmentation
        },
    }

    with open(os.path.join(yamldest, f"{datasetname}.yaml"), "w") as f:
        yaml.dump(data, f)


def create_mask_txt(txtpath, masks):
    """
    Converts a binary mask (PNG image) into the specified polygon format.

    Args:
        mask_path (str): Path to the binary mask PNG image.

    Returns:
        str: String representation in the specified polygon format (space-separated list of x,y coordinates).
    """

    with open(txtpath, "w") as f:  # Open the output file in write mode

        for m in masks:

            # Read the mask image in grayscale mode (assuming binary mask)
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            maxy, maxx = mask.shape

            if maxx > maxy:
                raise ValueError(f"Mask {m} width larger than height, unexpected!")

            # Find contours in the mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            if len(contours) > 1:
                RuntimeWarning("More than one contour detected in current mask")

            # If no contours are found, return an empty string (no object detected)
            if not contours:
                return ""

            # Get the first contour (assuming single object per mask)
            contour = contours[0]

            """# Approximate the contour with higher precision for smoother polygon 
            # higher epsilon for smoother curves with fewer points
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)"""

            # Convert the approximated polygon points to a list of x,y coordinates
            polygon_points = []
            for point in contour:
                x, y = point[0]
                polygon_points.append(str(x / maxx))
                polygon_points.append(str(y / maxy))

            # Format the output string with space-separated x,y coordinates
            polystring = " ".join(polygon_points)
            maskclass = 0
            if polystring:
                # Prepend class "0" and write to file with newline
                output_string = f"{maskclass} {polystring}\n"
                f.write(output_string)


def save_paths(jsonp, x_train0, y_train, x_val0, y_val, x_test0, y_test):
    x_train = [nameimg[1] for nameimg in x_train0]  # remove ear center name
    x_val = [nameimg[1] for nameimg in x_val0]

    pdict = {"x_train": x_train, "x_val": x_val, "y_train": y_train, "y_val": y_val}

    if TESTSET:
        x_test = [nameimg[1] for nameimg in x_test0]
        pdict["x_test"] = x_test
        pdict["y_test"] = y_test

    with open(f"{jsonp}.json", "w") as f:
        # Dump the data list as a JSON object
        json.dump(pdict, f)


def save_imgs_and_txts(x, y, dest, force_upright_imgs: bool = False, res_in_file_names: bool = False):
    for imgpname, maskps in zip(x, y):
        imgname, imgp = imgpname
        if res_in_file_names:
            if "5MP" in imgp:
                imgname = f"{imgname}_5MP"
            elif "20MP" in imgp:
                imgname = f"{imgname}_20MP"
            else:
                raise ValueError(f"Image {imgp} does not contain resolution info (5MP or 20MP) in path")
        txtpath = os.path.join(dest, f"{imgname}.txt")

        create_mask_txt(txtpath, maskps)
        if force_upright_imgs:
            img = cv2.imread(imgp)
            # check whether ear img x is larger than y (landscape)
            h, w = img.shape[:2]
            if w > h:
                print(f"Rotating {imgp} to {dest}")
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(os.path.join(dest, f"{imgname}.png"), img)
                continue
        shutil.copyfile(imgp, os.path.join(dest, f"{imgname}.png"))

def args():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO segmentation dataset from raw images and masks."
    )
    parser.add_argument(
        "--raw_img_path",
        type=str,
        required=True,
        help="Path to the directory containing raw images and masks.",
    )
    parser.add_argument(
        "--trainingdestbasep",
        type=str,
        required=True,
        help="Base path where the training dataset will be saved.",
    )
    parser.add_argument(
        "--datasetname",
        type=str,
        required=True,
        help="Name of the dataset (used for folder naming).",
    )
    parser.add_argument(
        "--train_test_split_json",
        type=str,
        default=None,
        help="Path to JSON file specifying train/test split ears.",
    )
    parser.add_argument(
        "--writesampaths",
        action="store_true",
        help="Flag to save sample paths as JSON files.",
    )
    parser.add_argument(
        "--writeyoloyaml",
        action="store_true",
        help="Flag to create YOLO YAML configuration file.",
    )
    parser.add_argument(
        "--ignoreups",
        action="store_true",
        help="Flag to ignore images with '_up_' in their names.",
    )
    parser.add_argument(
        "--testset",
        action="store_true",
        help="Flag to create a test set from the training data.",
    )
    parser.add_argument(
        "--force_upright_imgs",
        action="store_true",
        help="Flag to force all images to be upright (IN a few cases 5mp original pngs were rotated horizontally).",
    )
    parser.add_argument(
        "--res_in_file_names",
        action="store_true",
        help="Whether to include resolution info (5 MP or 20 MP) in image file names and mask file names (default False).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = args()
    WRITESAMPATHS = args.writesampaths#False
    WRITEYOLOYAML = args.writeyoloyaml#True
    IGNOREUPS = args.ignoreups#True
    TESTSET = args.testset#True

    raw_img_path = args.raw_img_path#"/media/geink81/hddE/3DMais/Ears/LabelData/EarsBreeders/20MP/PublicationTrainingsMasks"
    trainingdestbasep = args.trainingdestbasep#"/home/geink81/Mais/Ears/trainingdata"
    datasetname = args.datasetname#"PublicationTraining20MP_TVT"

    destfolder = os.path.join(trainingdestbasep, datasetname)
    trainfolder = os.path.join(destfolder, "train")
    valfolder = os.path.join(destfolder, "val")
    testfolder = os.path.join(destfolder, "test")

    if args.train_test_split_json:
        with open(args.train_test_split_json, 'r') as f:
            split_data = json.load(f)
        test_ears = split_data.get("test", [])
        trainval_ears = split_data.get("trainval", [])

        imgpsnames, maskps = prep_paths(raw_img_path, ears_to_skip=test_ears)
        imgpsnames_test, maskps_test = prep_paths(raw_img_path, ears_to_skip=trainval_ears)
        split_size = 0.25
    else:
        imgpsnames, maskps = prep_paths(raw_img_path)
        split_size = 0.2

    xtest, y_test = None, None

    # 60% training, 20% validation, 20% test
    x_train, x_val, y_train, y_val = train_test_split(
        imgpsnames, maskps, test_size=split_size, random_state=42
    )
    if TESTSET:
        if args.train_test_split_json:
            x_test, y_test = imgpsnames_test, maskps_test
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_train, y_train, test_size=0.25, random_state=42
            )
        print("TRAIN: ", len(x_train), "VAL: ", len(x_val), "TEST: ", len(x_test))
    else:
        print("TRAIN: ", len(x_train), "VAL: ", len(x_val))

    if (
        input(
            "Make sure no data will be overwritten if not intended. Continue? (y/n): "
        )
        != "y"
    ):
        raise ValueError("User aborted")

    if WRITESAMPATHS:
        samtrainingps = "SAMtrainingPaths"
        jsonp = os.path.join(samtrainingps, datasetname)
        if not os.path.exists(samtrainingps):
            os.mkdir(samtrainingps)
        save_paths(jsonp, x_train, y_train, x_val, y_val, x_test, y_test)

    if WRITEYOLOYAML:
        if not os.path.exists(trainingdestbasep):
            os.mkdir(trainingdestbasep)
        if not os.path.exists(destfolder):
            os.mkdir(destfolder)
        if not os.path.exists(trainfolder):
            os.mkdir(trainfolder)
        if not os.path.exists(valfolder):
            os.mkdir(valfolder)
        if not os.path.exists(testfolder) and TESTSET:
            os.mkdir(testfolder)

        save_imgs_and_txts(x_train, y_train, trainfolder, args.force_upright_imgs, args.res_in_file_names)
        save_imgs_and_txts(x_val, y_val, valfolder, args.force_upright_imgs, args.res_in_file_names)
        if TESTSET:
            save_imgs_and_txts(x_test, y_test, testfolder, args.force_upright_imgs, args.res_in_file_names)

        create_yaml_file(destfolder)
