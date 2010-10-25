from kapteyn import maputils
from matplotlib import pyplot as plt

# In the comments we show how to set a smaller box t display
f_red = maputils.FITSimage('m101_red.fits')
#f_red.set_limits((200,300),(200,300))
f_green = maputils.FITSimage('m101_green.fits')
#f_green.set_limits((200,300),(200,300))
f_blue = maputils.FITSimage('m101_blue.fits')
#f_blue.set_limits((200,300),(200,300))

# Show the three components R,G & B separately
# Show Z values when moving the mouse
fig = plt.figure()
frame = fig.add_subplot(2,2,1); frame.set_title("Red with noise")
a = f_red.Annotatedimage(frame); a.Image()
a.interact_toolbarinfo(wcsfmt=None, zfmt="%g")
frame = fig.add_subplot(2,2,2); frame.set_title("Greens are 1")
a = f_green.Annotatedimage(frame); a.Image()
a.interact_toolbarinfo(wcsfmt=None, zfmt="%g")
frame = fig.add_subplot(2,2,3); frame.set_title("Blues are 1")
a = f_blue.Annotatedimage(frame); a.Image()
a.interact_toolbarinfo(wcsfmt=None, zfmt="%g")

# Plot the composed RGB image
frame = fig.add_subplot(2,2,4); frame.set_title("RGB composed of previous")
annim = f_red.Annotatedimage(frame)
annim.RGBimage(f_red, f_green, f_blue, fun=lambda x:x*x, alpha=1)


# Note: color interaction not possible (RGB is fixed)
annim.interact_toolbarinfo(wcsfmt=None, zfmt="%g")
# Write RGB values to terminal after clicking left mouse button
annim.interact_writepos(pixfmt=None, wcsfmt="%.12f", zfmt="%.3e", hmsdms=False)
maputils.showall()