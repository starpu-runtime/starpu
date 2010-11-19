:
if ! libtool --version > /dev/null
then
	echo "Libtool is missing, please install it."
	exit 1
fi
autoreconf -ivf -I m4

