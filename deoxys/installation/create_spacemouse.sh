#!/usr/bin/env bash

set -euo pipefail

sudo tee /etc/udev/rules.d/77-spacemouse.rules >/dev/null <<'EOF'
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c62e", MODE="0660", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c62e", MODE="0660", GROUP="plugdev"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

echo "SpaceMouse udev rules installed."
echo "If your user was just added to plugdev, start a new shell or run: newgrp plugdev"
