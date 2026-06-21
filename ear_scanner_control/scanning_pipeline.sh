#!/usr/bin/env bash
set -euo pipefail

# simple venv builder + runner
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$DIR/venv"
REQ_FILE="$DIR/requirements.txt"

# first arg = python script to run (optional), rest are passed to that script
if [ $# -ge 1 ]; then
    PY_SCRIPT="$1"
    shift
else
    PY_SCRIPT="$DIR/stable_scan1500.py"
fi

INITIAL_RUN=false
# create venv if missing
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating python virtual environment..."
    INITIAL_RUN=true
    if command -v python3 >/dev/null 2>&1; then
        python3 -m venv "$VENV_DIR"
    elif command -v python >/dev/null 2>&1; then
        python -m venv "$VENV_DIR"
    else
        echo "No python interpreter found in PATH" >&2
        exit 1
    fi
fi

# activate venv 
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Unable to find venv activation script" >&2
    exit 1
fi


if [ -x "$VENV_DIR/bin/python" ]; then
    echo "Looking for apriltag build directory to add to venv site-packages..."
    SITE_PACKAGES="$($VENV_DIR/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
    PTH_FILE="$SITE_PACKAGES/global_packages.pth"

    # only act if the pth doesn't already mention apriltag
    if [ ! -f "$PTH_FILE" ] || ! grep -q "apriltag" "$PTH_FILE" 2>/dev/null; then
        # look for apriltag dir in the two specified locations
        APR_DIR=""
        if [ -d "$HOME/Desktop/apriltag" ]; then
            APR_DIR="$HOME/Desktop/apriltag"
        elif [ -d "$HOME/pythonstuff/apriltag" ]; then
            APR_DIR="$HOME/pythonstuff/apriltag"
        else
            # try a shallow search under those roots as a fallback
            APR_DIR=$(find "$HOME/Desktop" "$HOME/pythonstuff" -maxdepth 2 -type d -name apriltag 2>/dev/null | head -n1 || true)
        fi

        if [ -n "$APR_DIR" ]; then
            BUILD_PATH="$APR_DIR/build"
            if [ -d "$BUILD_PATH" ]; then
                mkdir -p "$(dirname "$PTH_FILE")"
                # append only if not already present
                if [ -f "$PTH_FILE" ]; then
                    if ! grep -Fxq "$BUILD_PATH" "$PTH_FILE" 2>/dev/null; then
                        echo "$BUILD_PATH" >> "$PTH_FILE"
                    fi
                else
                    echo "$BUILD_PATH" > "$PTH_FILE"
                fi
            fi
        fi
    fi
fi
      
if [ "$INITIAL_RUN" = true ] ; then
    echo 'Installing pip dependencies...'
    # ensure pip up-to-date and install requirements if present
    pip install --upgrade pip setuptools wheel >/dev/null
    if [ -f "$REQ_FILE" ]; then
        pip install -r "$REQ_FILE"
    fi
fi

# diagnose camera devices
echo "Checking camera devices..."
DEVICES=(/dev/video0 /dev/video1)

READY_TO_SCAN=true
for dev in "${DEVICES[@]}"; do
  if [ -e "$dev" ]; then
    echo "Found $dev"
    echo "  Permissions: $(stat -c '%a %U:%G' "$dev")"
    if command -v v4l2-ctl >/dev/null 2>&1; then
      echo "  v4l2 info:"
      v4l2-ctl -d "$dev" --all | sed -n '1,20p'
    fi
    # set camera parameters
    v4l2-ctl -d /dev/video0 --set-ctrl red_balance=2467
    v4l2-ctl -d /dev/video0 --set-ctrl blue_balance=1690
    v4l2-ctl -d /dev/video0 --set-ctrl exposure_absolute=175
    v4l2-ctl -d /dev/video1 --set-ctrl red_balance=2500
    v4l2-ctl -d /dev/video1 --set-ctrl blue_balance=1691
    v4l2-ctl -d /dev/video1 --set-ctrl exposure_absolute=177
   
  else
    echo "Missing $dev"
    READY_TO_SCAN=false

  fi
done

if [ "$READY_TO_SCAN" = false ] ; then
    echo "One or both camera devices are missing. Please ensure cameras are connected correctly an can be found at /dev/video0 and /dev/video1."
    exit 1
fi

echo "Starting scanning pipeline..."
# run the python script with remaining args
exec python "$PY_SCRIPT" "$@"