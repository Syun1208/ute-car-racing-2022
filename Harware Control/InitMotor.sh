systemctl disable serial-getty@ttyS0
systemctl stop nvgetty
systemctl disable nvgetty
udevadm trigger