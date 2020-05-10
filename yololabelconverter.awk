#!/usr/bin/awk -f

BEGIN {
	FS = " "
	width = 1920
	height = 1080
}
{
	x1 = $2
	y1 = $3
	x2 = $4
	y2 = $5
	class = $1

	# https://github.com/Guanghan/darknet/blob/master/scripts/convert.py
	# box = (float(xmin) 0, float(xmax) 1, float(ymin) 2, float(ymax) 3)
	dw = 1./width
	dh = 1./height
	x = (x1 + x2)/2.0
	y = (y1 + y2)/2.0
	w = x2 - x1
	h = y2 - y1
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh

	printf ("%s %f %f %f %f\n", class, x, y, w, h)
}
END {
	}
