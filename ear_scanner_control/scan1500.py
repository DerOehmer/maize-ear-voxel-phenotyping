from time import sleep, time
from v4l2py import Device
import numpy as np
import cv2
import os
import argparse
from talker import Talker
import threading


def buffer_to_img(byte_stream, destname):
    bytearr = np.frombuffer(byte_stream.data, dtype=np.uint8)
    argb_img = np.reshape(bytearr, (1944, 2592, 4))
    rgb_img = argb_img[:, :, 1:]
    img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imwrite(destname, img_rot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Enter input")
    parser.add_argument("-f", "--foldername")
    parser.add_argument("-r", "--rootfolder")
    parser.add_argument("-v", "--verbose", action="store_true")
    cobname = parser.parse_args().foldername
    rootfolder = parser.parse_args().rootfolder
    verbose = parser.parse_args().verbose

    start_time = time()
    N_IMAGES = 100
    STEPSFULLROT = 3200
    FTYPE = ".jpg"
    n_positions = int(N_IMAGES // 2)
    steps_per_move = int(STEPSFULLROT // n_positions)
    print("Steps per move: ", steps_per_move)

    img_folder = os.path.join(rootfolder, cobname)
    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"Invalid image folder: {img_folder}")

    SERIAL_PORT = "/dev/ttyACM0"

    t = Talker(SERIAL_PORT)
    t.send("enable_motor()")

    steps_done = 0
    pos_done = 1
    img_start = time()
    rottimestamp = 0

    with Device.from_id(0) as lowercam, Device.from_id(1) as uppercam:
        print(lowercam.controls.exposure_absolute)
        print(uppercam.controls.exposure_absolute)

        for i, (lowerframe, upperframe) in enumerate(zip(lowercam, uppercam)):

            ts = lowerframe.timestamp
            if i == 0:
                lowertimestampoffset = lowerframe.timestamp
                uppertimestampoffset = upperframe.timestamp
                reftimestart = time()

            elif i > 0:
                ###
                lowerframetimestamp = lowerframe.timestamp - lowertimestampoffset
                upperframetimestamp = upperframe.timestamp - uppertimestampoffset
                lowerdelay = rottimestamp - lowerframetimestamp
                upperdelay = rottimestamp - upperframetimestamp
                ###
                if (
                    lowerframetimestamp < rottimestamp
                    or upperframetimestamp < rottimestamp
                ):
                    if verbose:
                        print(
                            f"{lowerdelay} and {upperdelay} s delay: skippping current frame"
                        )
                    continue
                print(f"#### Position {pos_done} of {n_positions} ####")
                if steps_done > 0:
                    tlow.join()
                    tup.join()

                low_dest = f"{img_folder}/{cobname}_low_{steps_done}{FTYPE}"
                up_dest = f"{img_folder}/{cobname}_up_{steps_done}{FTYPE}"
                tlow = threading.Thread(
                    target=buffer_to_img, args=(lowerframe, low_dest)
                )
                tup = threading.Thread(target=buffer_to_img, args=(upperframe, up_dest))
                tlow.start()
                tup.start()

                t.send(f"stepping({steps_per_move})")

                while t.receive() != "done":
                    if verbose:
                        print("waiting for rotation to finish")
                    sleep(0.5)

                rottimestamp = time() - reftimestart + 0.4

                steps_done += steps_per_move
                pos_done += 1

                print(f"Capture procedure took {time()-img_start} s")
                img_start = time()
                if not steps_done < STEPSFULLROT:
                    break

        print(
            "Lower cam temperature: ",
            lowercam.controls.device_temperature.value / 10,
            " C",
        )  # max 85
        print(
            "Upper cam temperature: ",
            uppercam.controls.device_temperature.value / 10,
            " C",
        )

        tlow.join()
        tup.join()

    t.send("disable_motor()")
    print(
        f"Current scan including {N_IMAGES} images took {round(time() - start_time, 2)} seconds"
    )
