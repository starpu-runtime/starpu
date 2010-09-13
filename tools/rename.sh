find . -type f -not -path "*svn*"|xargs sed -i -f $(dirname $0)/rename.sed

